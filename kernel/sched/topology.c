// SPDX-License-Identifier: GPL-2.0
/*
 * Scheduler topology setup/handling methods
 */

/*
 * 调度拓扑阅读导航：
 *
 * 本文件把架构提供的 CPU 拓扑转换为调度器的负载均衡层级，负责构建、
 * 裁剪、发布和销毁拓扑对象；实际的选核与公平类负载均衡策略见 fair.c。
 *
 * 主要对象：
 *   sched_domain：每 CPU 一条 parent/child 链，span 随层级升高而扩大。
 *   sched_group：某层域内的横向分组，通常对应下一层子域，组成循环链表。
 *   sched_group_capacity：相同分组共享的容量数据，以引用计数管理生命周期。
 *   root_domain：一个调度分区的共享状态，包含 RT/DL 记账及 EAS 性能域。
 *
 * 主要路径：
 *   启动：sched_init_domains() -> build_sched_domains()。
 *   重建：partition_sched_domains() 对比新旧分区，仅替换需要变化的部分。
 *   构建：build_sched_domain() 建纵向域链，再构建横向组并初始化容量，
 *         最后由 cpu_attach_domain() 去掉冗余层级并挂接到运行队列。
 *   NUMA：sched_init_numa() 将节点距离转换成由近及远的累积 CPU 掩码。
 *   EAS：build_perf_domains() 将能耗模型关联到分区的 root_domain。
 *
 * 普通拓扑的同层分组互不相交；NUMA 分组可以重叠，需要额外的 balance
 * mask 确定哪些 CPU 能代表该组继续向上均衡。不要混淆组的 span 与此掩码。
 *
 * 拓扑重建由 CPU 热插拔锁与 sched_domains_mutex 等约束串行化；调度热路径
 * 通过 RCU 读取域指针；替换后需等待旧读者退出，再按共享对象的
 * 引用计数决定最终释放时机。
 */

#include <linux/sched/isolation.h>
#include <linux/bsearch.h>
#include "sched.h"

/* 串行化调度域重建，也保护构建过程中复用的临时 CPU 掩码。 */
DEFINE_MUTEX(sched_domains_mutex);
void sched_domains_mutex_lock(void)
{
	mutex_lock(&sched_domains_mutex);
}
void sched_domains_mutex_unlock(void)
{
	mutex_unlock(&sched_domains_mutex);
}

/* Protected by sched_domains_mutex: */
static cpumask_var_t sched_domains_tmpmask;
static cpumask_var_t sched_domains_tmpmask2;

/* 启动参数 sched_verbose 开启域构建及一致性检查的详细输出。 */
static int __init sched_debug_setup(char *str)
{
	sched_debug_verbose = true;

	return 0;
}
early_param("sched_verbose", sched_debug_setup);

static inline bool sched_debug(void)
{
	return sched_debug_verbose;
}

#define SD_FLAG(_name, mflags) [__##_name] = { .meta_flags = mflags, .name = #_name },
const struct sd_flag_debug sd_flag_debug[] = {
#include <linux/sched/sd_flags.h>
};
#undef SD_FLAG

/*
 * 打印单层域，检查 CPU 归属、标志继承、组覆盖及父子包含关系。
 * groupmask 是复用的临时掩码，用于累计本域所有组覆盖的 CPU。
 */
static int sched_domain_debug_one(struct sched_domain *sd, int cpu, int level,
				  struct cpumask *groupmask)
{
	struct sched_group *group = sd->groups;
	unsigned long flags = sd->flags;
	unsigned int idx;

	cpumask_clear(groupmask);

	printk(KERN_DEBUG "%*s domain-%d: ", level, "", level);
	printk(KERN_CONT "span=%*pbl level=%s\n",
	       cpumask_pr_args(sched_domain_span(sd)), sd->name);

	if (!cpumask_test_cpu(cpu, sched_domain_span(sd))) {
		printk(KERN_ERR "ERROR: domain->span does not contain CPU%d\n", cpu);
	}
	if (group && !cpumask_test_cpu(cpu, sched_group_span(group))) {
		printk(KERN_ERR "ERROR: domain->groups does not contain CPU%d\n", cpu);
	}

	/* 标志元数据描述其是否必须同时出现在子域或父域。 */
	for_each_set_bit(idx, &flags, __SD_FLAG_CNT) {
		unsigned int flag = BIT(idx);
		unsigned int meta_flags = sd_flag_debug[idx].meta_flags;

		if ((meta_flags & SDF_SHARED_CHILD) && sd->child &&
		    !(sd->child->flags & flag))
			printk(KERN_ERR "ERROR: flag %s set here but not in child\n",
			       sd_flag_debug[idx].name);

		if ((meta_flags & SDF_SHARED_PARENT) && sd->parent &&
		    !(sd->parent->flags & flag))
			printk(KERN_ERR "ERROR: flag %s set here but not in parent\n",
			       sd_flag_debug[idx].name);
	}

	printk(KERN_DEBUG "%*s groups:", level + 1, "");
	do {
		if (!group) {
			printk("\n");
			printk(KERN_ERR "ERROR: group is NULL\n");
			break;
		}

		if (cpumask_empty(sched_group_span(group))) {
			printk(KERN_CONT "\n");
			printk(KERN_ERR "ERROR: empty group\n");
			break;
		}

		if (!(sd->flags & SD_NUMA) &&
		    cpumask_intersects(groupmask, sched_group_span(group))) {
			printk(KERN_CONT "\n");
			printk(KERN_ERR "ERROR: repeated CPUs\n");
			break;
		}

		cpumask_or(groupmask, groupmask, sched_group_span(group));

		printk(KERN_CONT " %d:{ span=%*pbl",
				group->sgc->id,
				cpumask_pr_args(sched_group_span(group)));

		if ((sd->flags & SD_NUMA) &&
		    !cpumask_equal(group_balance_mask(group), sched_group_span(group))) {
			printk(KERN_CONT " mask=%*pbl",
				cpumask_pr_args(group_balance_mask(group)));
		}

		if (group->sgc->capacity != SCHED_CAPACITY_SCALE)
			printk(KERN_CONT " cap=%lu", group->sgc->capacity);

		if (group == sd->groups && sd->child &&
		    !cpumask_equal(sched_domain_span(sd->child),
				   sched_group_span(group))) {
			printk(KERN_ERR "ERROR: domain->groups does not match domain->child\n");
		}

		printk(KERN_CONT " }");

		group = group->next;

		if (group != sd->groups)
			printk(KERN_CONT ",");

	} while (group != sd->groups);
	printk(KERN_CONT "\n");

	if (!cpumask_equal(sched_domain_span(sd), groupmask))
		printk(KERN_ERR "ERROR: groups don't span domain->span\n");

	if (sd->parent &&
	    !cpumask_subset(groupmask, sched_domain_span(sd->parent)))
		printk(KERN_ERR "ERROR: parent span is not a superset of domain->span\n");
	return 0;
}

/* 从 CPU 的最底层域向上打印完整域链；未开启详细调试时直接返回。 */
static void sched_domain_debug(struct sched_domain *sd, int cpu)
{
	int level = 0;

	if (!sched_debug_verbose)
		return;

	if (!sd) {
		printk(KERN_DEBUG "CPU%d attaching NULL sched-domain.\n", cpu);
		return;
	}

	printk(KERN_DEBUG "CPU%d attaching sched-domain(s):\n", cpu);

	for (;;) {
		if (sched_domain_debug_one(sd, cpu, level, sched_domains_tmpmask))
			break;
		level++;
		sd = sd->parent;
		if (!sd)
			break;
	}
}

/* Generate a mask of SD flags with the SDF_NEEDS_GROUPS metaflag */
#define SD_FLAG(name, mflags) (name * !!((mflags) & SDF_NEEDS_GROUPS)) |
static const unsigned int SD_DEGENERATE_GROUPS_MASK =
#include <linux/sched/sd_flags.h>
0;
#undef SD_FLAG

/*
 * 判断单个域是否已无调度价值：只有一个 CPU，或既没有可用的组间操作，
 * 也没有 SD_WAKE_AFFINE 等不依赖组的功能时，该域可以被裁剪。
 */
static int sd_degenerate(struct sched_domain *sd)
{
	if (cpumask_weight(sched_domain_span(sd)) == 1)
		return 1;

	/* Following flags need at least 2 groups */
	if ((sd->flags & SD_DEGENERATE_GROUPS_MASK) &&
	    (sd->groups != sd->groups->next))
		return 0;

	/* Following flags don't use groups */
	if (sd->flags & (SD_WAKE_AFFINE))
		return 0;

	return 1;
}

/*
 * 判断父域是否冗余：自身退化，或与子域范围相同且没有新增有效标志，
 * 均无需保留这一层；单组父域的组间操作标志不计入比较。
 */
static int
sd_parent_degenerate(struct sched_domain *sd, struct sched_domain *parent)
{
	unsigned long cflags = sd->flags, pflags = parent->flags;

	if (sd_degenerate(parent))
		return 1;

	if (!cpumask_equal(sched_domain_span(sd), sched_domain_span(parent)))
		return 0;

	/* Flags needing groups don't count if only 1 group in parent */
	if (parent->groups == parent->groups->next)
		pflags &= ~SD_DEGENERATE_GROUPS_MASK;

	if (~cflags & pflags)
		return 0;

	return 1;
}

#if defined(CONFIG_ENERGY_MODEL) && defined(CONFIG_CPU_FREQ_GOV_SCHEDUTIL)
DEFINE_STATIC_KEY_FALSE(sched_energy_present);
static unsigned int sysctl_sched_energy_aware = 1;
static DEFINE_MUTEX(sched_energy_mutex);
static bool sched_energy_update;

/*
 * 检查分区是否具备 EAS 的拓扑、负载跟踪和调频条件。
 * 此处不创建性能域，也不验证每个 CPU 的能耗模型；后者由 pd_init() 检查。
 */
static bool sched_is_eas_possible(const struct cpumask *cpu_mask)
{
	bool any_asym_capacity = false;
	int i;

	/* EAS is enabled for asymmetric CPU capacity topologies. */
	for_each_cpu(i, cpu_mask) {
		if (rcu_access_pointer(per_cpu(sd_asym_cpucapacity, i))) {
			any_asym_capacity = true;
			break;
		}
	}
	if (!any_asym_capacity) {
		if (sched_debug()) {
			pr_info("rd %*pbl: Checking EAS, CPUs do not have asymmetric capacities\n",
				cpumask_pr_args(cpu_mask));
		}
		return false;
	}

	/* EAS definitely does *not* handle SMT */
	if (sched_smt_active()) {
		if (sched_debug()) {
			pr_info("rd %*pbl: Checking EAS, SMT is not supported\n",
				cpumask_pr_args(cpu_mask));
		}
		return false;
	}

	if (!arch_scale_freq_invariant()) {
		if (sched_debug()) {
			pr_info("rd %*pbl: Checking EAS: frequency-invariant load tracking not yet supported",
				cpumask_pr_args(cpu_mask));
		}
		return false;
	}

	if (!cpufreq_ready_for_eas(cpu_mask)) {
		if (sched_debug()) {
			pr_info("rd %*pbl: Checking EAS: cpufreq is not ready\n",
				cpumask_pr_args(cpu_mask));
		}
		return false;
	}

	return true;
}

/*
 * 强制刷新 EAS 性能域，即使 CPU 分区没有变化也不能复用旧的性能域链。
 * 独立互斥锁保护 sched_energy_update 覆盖的整个同步重建过程。
 */
void rebuild_sched_domains_energy(void)
{
	mutex_lock(&sched_energy_mutex);
	sched_energy_update = true;
	rebuild_sched_domains();
	sched_energy_update = false;
	mutex_unlock(&sched_energy_mutex);
}

#ifdef CONFIG_PROC_SYSCTL
/* EAS sysctl：校验写权限和平台能力，开关变化时重建相关状态。 */
static int sched_energy_aware_handler(const struct ctl_table *table, int write,
		void *buffer, size_t *lenp, loff_t *ppos)
{
	int ret, state;

	if (write && !capable(CAP_SYS_ADMIN))
		return -EPERM;

	if (!sched_is_eas_possible(cpu_active_mask)) {
		if (write) {
			return -EOPNOTSUPP;
		} else {
			*lenp = 0;
			return 0;
		}
	}

	ret = proc_dointvec_minmax(table, write, buffer, lenp, ppos);
	if (!ret && write) {
		state = static_branch_unlikely(&sched_energy_present);
		if (state != sysctl_sched_energy_aware)
			rebuild_sched_domains_energy();
	}

	return ret;
}

static const struct ctl_table sched_energy_aware_sysctls[] = {
	{
		.procname       = "sched_energy_aware",
		.data           = &sysctl_sched_energy_aware,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = sched_energy_aware_handler,
		.extra1         = SYSCTL_ZERO,
		.extra2         = SYSCTL_ONE,
	},
};

static int __init sched_energy_aware_sysctl_init(void)
{
	register_sysctl_init("kernel", sched_energy_aware_sysctls);
	return 0;
}

late_initcall(sched_energy_aware_sysctl_init);
#endif /* CONFIG_PROC_SYSCTL */

/* 仅释放调度器的性能域链表节点；其引用的 EM 对象不归本函数所有。 */
static void free_pd(struct perf_domain *pd)
{
	struct perf_domain *tmp;

	while (pd) {
		tmp = pd->next;
		kfree(pd);
		pd = tmp;
	}
}

/* 按 CPU 查找已覆盖它的性能域，避免为同一个 EM 性能域重复建节点。 */
static struct perf_domain *find_pd(struct perf_domain *pd, int cpu)
{
	while (pd) {
		if (cpumask_test_cpu(cpu, perf_domain_span(pd)))
			return pd;
		pd = pd->next;
	}

	return NULL;
}

/* 基于 CPU 的能耗模型创建封装；缺少 EM 或分配失败返回 NULL。 */
static struct perf_domain *pd_init(int cpu)
{
	struct em_perf_domain *obj = em_cpu_get(cpu);
	struct perf_domain *pd;

	if (!obj) {
		if (sched_debug())
			pr_info("%s: no EM found for CPU%d\n", __func__, cpu);
		return NULL;
	}

	pd = kzalloc(sizeof(*pd), GFP_KERNEL);
	if (!pd)
		return NULL;
	pd->em_pd = obj;

	return pd;
}

/* 调试输出分区关联的性能域、CPU 范围及性能状态数量。 */
static void perf_domain_debug(const struct cpumask *cpu_map,
						struct perf_domain *pd)
{
	if (!sched_debug() || !pd)
		return;

	printk(KERN_DEBUG "root_domain %*pbl:", cpumask_pr_args(cpu_map));

	while (pd) {
		printk(KERN_CONT " pd%d:{ cpus=%*pbl nr_pstate=%d }",
				cpumask_first(perf_domain_span(pd)),
				cpumask_pr_args(perf_domain_span(pd)),
				em_pd_nr_perf_states(pd->em_pd));
		pd = pd->next;
	}

	printk(KERN_CONT "\n");
}

/* RCU 宽限期结束后，旧性能域链已不再被选核路径使用，可以整体回收。 */
static void destroy_perf_domain_rcu(struct rcu_head *rp)
{
	struct perf_domain *pd;

	pd = container_of(rp, struct perf_domain, rcu);
	free_pd(pd);
}

/* 只要任一分区支持 EAS，就开启全局静态分支；最后一个退出时再关闭。 */
static void sched_energy_set(bool has_eas)
{
	if (!has_eas && static_branch_unlikely(&sched_energy_present)) {
		if (sched_debug())
			pr_info("%s: stopping EAS\n", __func__);
		static_branch_disable_cpuslocked(&sched_energy_present);
	} else if (has_eas && !static_branch_unlikely(&sched_energy_present)) {
		if (sched_debug())
			pr_info("%s: starting EAS\n", __func__);
		static_branch_enable_cpuslocked(&sched_energy_present);
	}
}

/*
 * EAS can be used on a root domain if it meets all the following conditions:
 *    1. an Energy Model (EM) is available;
 *    2. the SD_ASYM_CPUCAPACITY flag is set in the sched_domain hierarchy.
 *    3. no SMT is detected.
 *    4. schedutil is driving the frequency of all CPUs of the rd;
 *    5. frequency invariance support is present;
 *
 * 为一个 root_domain 构建性能域链，先在私有链表上完成，再通过 RCU 替换。
 * 性能域的范围来自 EM，可能跨越调度分区边界，并非按 cpu_map 重新切割。
 * 任何一项检查或分配失败都会清除该 root_domain 的 EAS 性能域关联。
 */
static bool build_perf_domains(const struct cpumask *cpu_map)
{
	int i;
	struct perf_domain *pd = NULL, *tmp;
	int cpu = cpumask_first(cpu_map);
	struct root_domain *rd = cpu_rq(cpu)->rd;

	if (!sysctl_sched_energy_aware)
		goto free;

	if (!sched_is_eas_possible(cpu_map))
		goto free;

	for_each_cpu(i, cpu_map) {
		/* Skip already covered CPUs. */
		if (find_pd(pd, i))
			continue;

		/* Create the new pd and add it to the local list. */
		tmp = pd_init(i);
		if (!tmp)
			goto free;
		tmp->next = pd;
		pd = tmp;
	}

	perf_domain_debug(cpu_map, pd);

	/* Attach the new list of performance domains to the root domain. */
	tmp = rd->pd;
	rcu_assign_pointer(rd->pd, pd);
	if (tmp)
		call_rcu(&tmp->rcu, destroy_perf_domain_rcu);

	return !!pd;

free:
	/* 新链未发布，可直接释放；旧链可能仍有读者，需延迟回收。 */
	free_pd(pd);
	tmp = rd->pd;
	rcu_assign_pointer(rd->pd, NULL);
	if (tmp)
		call_rcu(&tmp->rcu, destroy_perf_domain_rcu);

	return false;
}
#else /* !(CONFIG_ENERGY_MODEL && CONFIG_CPU_FREQ_GOV_SCHEDUTIL): */
static void free_pd(struct perf_domain *pd) { }
#endif /* !(CONFIG_ENERGY_MODEL && CONFIG_CPU_FREQ_GOV_SCHEDUTIL) */

/*
 * 释放根域的掩码和 RT/DL/EAS 状态。已发布对象需先引用归零并等待 RCU；
 * 构建失败、从未挂接运行队列的根域则可直接调用本函数清理。
 */
static void free_rootdomain(struct rcu_head *rcu)
{
	struct root_domain *rd = container_of(rcu, struct root_domain, rcu);

	cpupri_cleanup(&rd->cpupri);
	cpudl_cleanup(&rd->cpudl);
	free_cpumask_var(rd->dlo_mask);
	free_cpumask_var(rd->rto_mask);
	free_cpumask_var(rd->online);
	free_cpumask_var(rd->span);
	free_pd(rd->pd);
	kfree(rd);
}

/*
 * 将运行队列迁移到新的 root_domain。持 rq 锁更新 CPU 归属、在线状态和
 * 调度类记账；旧 root_domain 的最后一个引用退出后通过 RCU 回收。
 */
void rq_attach_root(struct rq *rq, struct root_domain *rd)
{
	struct root_domain *old_rd = NULL;
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);

	/* 从旧分区撤销在线记账，再移除成员身份和运行队列引用。 */
	if (rq->rd) {
		old_rd = rq->rd;

		if (cpumask_test_cpu(rq->cpu, old_rd->online))
			set_rq_offline(rq);

		cpumask_clear_cpu(rq->cpu, old_rd->span);

		/*
		 * If we don't want to free the old_rd yet then
		 * set old_rd to NULL to skip the freeing later
		 * in this function:
		 */
		if (!atomic_dec_and_test(&old_rd->refcount))
			old_rd = NULL;
	}

	/* span 记录成员；仅 active CPU 在新分区中恢复在线记账。 */
	atomic_inc(&rd->refcount);
	rq->rd = rd;

	cpumask_set_cpu(rq->cpu, rd->span);
	if (cpumask_test_cpu(rq->cpu, cpu_active_mask))
		set_rq_online(rq);

	/*
	 * Because the rq is not a task, dl_add_task_root_domain() did not
	 * move the fair server bw to the rd if it already started.
	 * Add it now.
	 */
	if (rq->fair_server.dl_server)
		__dl_server_attach_root(&rq->fair_server, rq);

	rq_unlock_irqrestore(rq, &rf);

	if (old_rd)
		call_rcu(&old_rd->rcu, free_rootdomain);
}

/* 额外持有 root_domain 引用，避免它随运行队列脱离而被回收。 */
void sched_get_rd(struct root_domain *rd)
{
	atomic_inc(&rd->refcount);
}

/* 释放引用；归零后通过 RCU 延迟销毁，保护现有读者。 */
void sched_put_rd(struct root_domain *rd)
{
	if (!atomic_dec_and_test(&rd->refcount))
		return;

	call_rcu(&rd->rcu, free_rootdomain);
}

/*
 * 初始化分区 CPU 掩码、DL 带宽/截止期索引及 RT 优先级索引。
 * 失败时按分配的逆序回滚；对象本身由调用者负责释放。
 */
static int init_rootdomain(struct root_domain *rd)
{
	if (!zalloc_cpumask_var(&rd->span, GFP_KERNEL))
		goto out;
	if (!zalloc_cpumask_var(&rd->online, GFP_KERNEL))
		goto free_span;
	if (!zalloc_cpumask_var(&rd->dlo_mask, GFP_KERNEL))
		goto free_online;
	if (!zalloc_cpumask_var(&rd->rto_mask, GFP_KERNEL))
		goto free_dlo_mask;

#ifdef HAVE_RT_PUSH_IPI
	rd->rto_cpu = -1;
	raw_spin_lock_init(&rd->rto_lock);
	rd->rto_push_work = IRQ_WORK_INIT_HARD(rto_push_irq_work_func);
#endif

	rd->visit_cookie = 0;
	init_dl_bw(&rd->dl_bw);
	if (cpudl_init(&rd->cpudl) != 0)
		goto free_rto_mask;

	if (cpupri_init(&rd->cpupri) != 0)
		goto free_cpudl;
	return 0;

free_cpudl:
	cpudl_cleanup(&rd->cpudl);
free_rto_mask:
	free_cpumask_var(rd->rto_mask);
free_dlo_mask:
	free_cpumask_var(rd->dlo_mask);
free_online:
	free_cpumask_var(rd->online);
free_span:
	free_cpumask_var(rd->span);
out:
	return -ENOMEM;
}

/*
 * By default the system creates a single root-domain with all CPUs as
 * members (mimicking the global state we have today).
 */
struct root_domain def_root_domain;

/* 默认根域保留一个永久引用，供未挂接普通调度域的 CPU 等场景使用。 */
void __init init_defrootdomain(void)
{
	init_rootdomain(&def_root_domain);

	atomic_set(&def_root_domain.refcount, 1);
}

/* 创建尚未关联运行队列的新根域；挂接时才增加其运行队列引用。 */
static struct root_domain *alloc_rootdomain(void)
{
	struct root_domain *rd;

	rd = kzalloc(sizeof(*rd), GFP_KERNEL);
	if (!rd)
		return NULL;

	if (init_rootdomain(rd) != 0) {
		kfree(rd);
		return NULL;
	}

	return rd;
}

/*
 * 遍历组循环链表，减少组引用并在归零时释放。free_sgc 指明是否同时减少
 * 共享容量引用；构建失败时，仍归临时分配表管理的容量不在此释放。
 */
static void free_sched_groups(struct sched_group *sg, int free_sgc)
{
	struct sched_group *tmp, *first;

	if (!sg)
		return;

	first = sg;
	do {
		tmp = sg->next;

		if (free_sgc && atomic_dec_and_test(&sg->sgc->ref))
			kfree(sg->sgc);

		if (atomic_dec_and_test(&sg->ref))
			kfree(sg);
		sg = tmp;
	} while (sg != first);
}

/* 销毁单层域，按引用计数释放共享组、容量和 LLC 状态。 */
static void destroy_sched_domain(struct sched_domain *sd)
{
	/*
	 * A normal sched domain may have multiple group references, an
	 * overlapping domain, having private groups, only one.  Iterate,
	 * dropping group/capacity references, freeing where none remain.
	 */
	free_sched_groups(sd->groups, 1);

	if (sd->shared && atomic_dec_and_test(&sd->shared->ref))
		kfree(sd->shared);
	kfree(sd);
}

/* 最底层域发起的 RCU 回调：沿 parent 链逐层释放整条旧拓扑。 */
static void destroy_sched_domains_rcu(struct rcu_head *rcu)
{
	struct sched_domain *sd = container_of(rcu, struct sched_domain, rcu);

	while (sd) {
		struct sched_domain *parent = sd->parent;
		destroy_sched_domain(sd);
		sd = parent;
	}
}

/* 域链已从运行队列摘除，但仍需等待调度热路径的 RCU 读者退出。 */
static void destroy_sched_domains(struct sched_domain *sd)
{
	if (sd)
		call_rcu(&sd->rcu, destroy_sched_domains_rcu);
}

/*
 * Keep a special pointer to the highest sched_domain that has SD_SHARE_LLC set
 * (Last Level Cache Domain) for this allows us to avoid some pointer chasing
 * select_idle_sibling().
 *
 * Also keep a unique ID per domain (we use the first CPU number in the cpumask
 * of the domain), this allows us to quickly tell if two CPUs are in the same
 * cache domain, see cpus_share_cache().
 */
DEFINE_PER_CPU(struct sched_domain __rcu *, sd_llc);
DEFINE_PER_CPU(int, sd_llc_size);
DEFINE_PER_CPU(int, sd_llc_id);
DEFINE_PER_CPU(int, sd_share_id);
DEFINE_PER_CPU(struct sched_domain_shared __rcu *, sd_llc_shared);
DEFINE_PER_CPU(struct sched_domain __rcu *, sd_numa);
DEFINE_PER_CPU(struct sched_domain __rcu *, sd_asym_packing);
DEFINE_PER_CPU(struct sched_domain __rcu *, sd_asym_cpucapacity);

DEFINE_STATIC_KEY_FALSE(sched_asym_cpucapacity);
DEFINE_STATIC_KEY_FALSE(sched_cluster_active);

/*
 * 刷新每 CPU 的拓扑快捷指针和资源 ID，避免选核热路径遍历域链。
 * LLC/非对称打包取最高匹配层，NUMA/完整容量非对称域则取最低匹配层。
 */
static void update_top_cache_domain(int cpu)
{
	struct sched_domain_shared *sds = NULL;
	struct sched_domain *sd;
	int id = cpu;
	int size = 1;

	sd = highest_flag_domain(cpu, SD_SHARE_LLC);
	if (sd) {
		id = cpumask_first(sched_domain_span(sd));
		size = cpumask_weight(sched_domain_span(sd));
		sds = sd->shared;
	}

	rcu_assign_pointer(per_cpu(sd_llc, cpu), sd);
	per_cpu(sd_llc_size, cpu) = size;
	per_cpu(sd_llc_id, cpu) = id;
	rcu_assign_pointer(per_cpu(sd_llc_shared, cpu), sds);

	sd = lowest_flag_domain(cpu, SD_CLUSTER);
	if (sd)
		id = cpumask_first(sched_domain_span(sd));

	/*
	 * This assignment should be placed after the sd_llc_id as
	 * we want this id equals to cluster id on cluster machines
	 * but equals to LLC id on non-Cluster machines.
	 */
	per_cpu(sd_share_id, cpu) = id;

	sd = lowest_flag_domain(cpu, SD_NUMA);
	rcu_assign_pointer(per_cpu(sd_numa, cpu), sd);

	sd = highest_flag_domain(cpu, SD_ASYM_PACKING);
	rcu_assign_pointer(per_cpu(sd_asym_packing, cpu), sd);

	sd = lowest_flag_domain(cpu, SD_ASYM_CPUCAPACITY_FULL);
	rcu_assign_pointer(per_cpu(sd_asym_cpucapacity, cpu), sd);
}

/*
 * Attach the domain 'sd' to 'cpu' as its base domain. Callers must
 * hold the hotplug lock.
 *
 * 先裁剪没有额外调度价值的层级，再切换根域和 rq->sd，并更新快捷指针。
 * 未发布的冗余域可直接释放；被替换的旧域链必须经 RCU 回收。
 */
static void
cpu_attach_domain(struct sched_domain *sd, struct root_domain *rd, int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	struct sched_domain *tmp;

	/* Remove the sched domains which do not contribute to scheduling. */
	for (tmp = sd; tmp; ) {
		struct sched_domain *parent = tmp->parent;
		if (!parent)
			break;

		if (sd_parent_degenerate(tmp, parent)) {
			tmp->parent = parent->parent;

			if (parent->parent) {
				parent->parent->child = tmp;
				parent->parent->groups->flags = tmp->flags;
			}

			/*
			 * Transfer SD_PREFER_SIBLING down in case of a
			 * degenerate parent; the spans match for this
			 * so the property transfers.
			 */
			if (parent->flags & SD_PREFER_SIBLING)
				tmp->flags |= SD_PREFER_SIBLING;
			destroy_sched_domain(parent);
		} else
			tmp = tmp->parent;
	}

	if (sd && sd_degenerate(sd)) {
		tmp = sd;
		sd = sd->parent;
		destroy_sched_domain(tmp);
		if (sd) {
			struct sched_group *sg = sd->groups;

			/*
			 * sched groups hold the flags of the child sched
			 * domain for convenience. Clear such flags since
			 * the child is being destroyed.
			 */
			do {
				sg->flags = 0;
			} while (sg != sd->groups);

			sd->child = NULL;
		}
	}

	sched_domain_debug(sd, cpu);

	/* 新链完成初始化后才发布，防止 RCU 读者看到半构建的域结构。 */
	rq_attach_root(rq, rd);
	tmp = rq->sd;
	rcu_assign_pointer(rq->sd, sd);
	dirty_sched_domain_sysctl(cpu);
	destroy_sched_domains(tmp);

	update_top_cache_domain(cpu);
}

/* 一次分区构建的上下文：保存每 CPU 的底层域入口和该分区的新根域。 */
struct s_data {
	struct sched_domain * __percpu *sd;
	struct root_domain	*rd;
};

/* 清理阶段按分配进度逐层回收，统一处理构建失败和成功收尾。 */
enum s_alloc {
	sa_rootdomain,
	sa_sd,
	sa_sd_storage,
	sa_none,
};

/*
 * Return the canonical balance CPU for this group, this is the first CPU
 * of this group that's also in the balance mask.
 *
 * The balance mask are all those CPUs that could actually end up at this
 * group. See build_balance_mask().
 *
 * Also see should_we_balance().
 */
int group_balance_cpu(struct sched_group *sg)
{
	return cpumask_first(group_balance_mask(sg));
}


/*
 * NUMA topology (first read the regular topology blurb below)
 *
 * Given a node-distance table, for example:
 *
 *   node   0   1   2   3
 *     0:  10  20  30  20
 *     1:  20  10  20  30
 *     2:  30  20  10  20
 *     3:  20  30  20  10
 *
 * which represents a 4 node ring topology like:
 *
 *   0 ----- 1
 *   |       |
 *   |       |
 *   |       |
 *   3 ----- 2
 *
 * We want to construct domains and groups to represent this. The way we go
 * about doing this is to build the domains on 'hops'. For each NUMA level we
 * construct the mask of all nodes reachable in @level hops.
 *
 * For the above NUMA topology that gives 3 levels:
 *
 * NUMA-2	0-3		0-3		0-3		0-3
 *  groups:	{0-1,3},{1-3}	{0-2},{0,2-3}	{1-3},{0-1,3}	{0,2-3},{0-2}
 *
 * NUMA-1	0-1,3		0-2		1-3		0,2-3
 *  groups:	{0},{1},{3}	{0},{1},{2}	{1},{2},{3}	{0},{2},{3}
 *
 * NUMA-0	0		1		2		3
 *
 *
 * As can be seen; things don't nicely line up as with the regular topology.
 * When we iterate a domain in child domain chunks some nodes can be
 * represented multiple times -- hence the "overlap" naming for this part of
 * the topology.
 *
 * In order to minimize this overlap, we only build enough groups to cover the
 * domain. For instance Node-0 NUMA-2 would only get groups: 0-1,3 and 1-3.
 *
 * Because:
 *
 *  - the first group of each domain is its child domain; this
 *    gets us the first 0-1,3
 *  - the only uncovered node is 2, who's child domain is 1-3.
 *
 * However, because of the overlap, computing a unique CPU for each group is
 * more complicated. Consider for instance the groups of NODE-1 NUMA-2, both
 * groups include the CPUs of Node-0, while those CPUs would not in fact ever
 * end up at those groups (they would end up in group: 0-1,3).
 *
 * To correct this we have to introduce the group balance mask. This mask
 * will contain those CPUs in the group that can reach this group given the
 * (child) domain tree.
 *
 * With this we can once again compute balance_cpu and sched_group_capacity
 * relations.
 *
 * XXX include words on how balance_cpu is unique and therefore can be
 * used for sched_group_capacity links.
 *
 *
 * Another 'interesting' topology is:
 *
 *   node   0   1   2   3
 *     0:  10  20  20  30
 *     1:  20  10  20  20
 *     2:  20  20  10  20
 *     3:  30  20  20  10
 *
 * Which looks a little like:
 *
 *   0 ----- 1
 *   |     / |
 *   |   /   |
 *   | /     |
 *   2 ----- 3
 *
 * This topology is asymmetric, nodes 1,2 are fully connected, but nodes 0,3
 * are not.
 *
 * This leads to a few particularly weird cases where the sched_domain's are
 * not of the same number for each CPU. Consider:
 *
 * NUMA-2	0-3						0-3
 *  groups:	{0-2},{1-3}					{1-3},{0-2}
 *
 * NUMA-1	0-2		0-3		0-3		1-3
 *
 * NUMA-0	0		1		2		3
 *
 */


/*
 * Build the balance mask; it contains only those CPUs that can arrive at this
 * group and should be considered to continue balancing.
 *
 * We do this during the group creation pass, therefore the group information
 * isn't complete yet, however since each group represents a (child) domain we
 * can fully construct this using the sched_domain bits (which are already
 * complete).
 *
 * 重叠组的 span 只是成员集合，不代表其中每个 CPU 都能沿自己的域链进入
 * 此组。仅保留其同层域的 child 与本组 span 相同的 CPU，作为均衡候选。
 */
static void
build_balance_mask(struct sched_domain *sd, struct sched_group *sg, struct cpumask *mask)
{
	const struct cpumask *sg_span = sched_group_span(sg);
	struct sd_data *sdd = sd->private;
	struct sched_domain *sibling;
	int i;

	cpumask_clear(mask);

	for_each_cpu(i, sg_span) {
		sibling = *per_cpu_ptr(sdd->sd, i);

		/*
		 * Can happen in the asymmetric case, where these siblings are
		 * unused. The mask will not be empty because those CPUs that
		 * do have the top domain _should_ span the domain.
		 */
		if (!sibling->child)
			continue;

		/* If we would not end up here, we can't continue from here */
		if (!cpumask_equal(sg_span, sched_domain_span(sibling->child)))
			continue;

		cpumask_set_cpu(i, mask);
	}

	/* We must not have empty masks here */
	WARN_ON_ONCE(cpumask_empty(mask));
}

/*
 * XXX: This creates per-node group entries; since the load-balancer will
 * immediately access remote memory to construct this group's load-balance
 * statistics having the groups node local is of dubious benefit.
 *
 * 为重叠拓扑分配私有组，复制子域的 span 和标志，稍后绑定容量对象。
 */
static struct sched_group *
build_group_from_child_sched_domain(struct sched_domain *sd, int cpu)
{
	struct sched_group *sg;
	struct cpumask *sg_span;

	sg = kzalloc_node(sizeof(struct sched_group) + cpumask_size(),
			GFP_KERNEL, cpu_to_node(cpu));

	if (!sg)
		return NULL;

	sg_span = sched_group_span(sg);
	if (sd->child) {
		cpumask_copy(sg_span, sched_domain_span(sd->child));
		sg->flags = sd->child->flags;
	} else {
		cpumask_copy(sg_span, sched_domain_span(sd));
	}

	atomic_inc(&sg->ref);
	return sg;
}

/*
 * 用 balance mask 中的首个 CPU 作为共享容量对象的索引。
 * 重叠域的组节点虽各自独立，代表同一组的容量与 balance mask 仍应共享。
 */
static void init_overlap_sched_group(struct sched_domain *sd,
				     struct sched_group *sg)
{
	struct cpumask *mask = sched_domains_tmpmask2;
	struct sd_data *sdd = sd->private;
	struct cpumask *sg_span;
	int cpu;

	build_balance_mask(sd, sg, mask);
	cpu = cpumask_first(mask);

	sg->sgc = *per_cpu_ptr(sdd->sgc, cpu);
	if (atomic_inc_return(&sg->sgc->ref) == 1)
		cpumask_copy(group_balance_mask(sg), mask);
	else
		WARN_ON_ONCE(!cpumask_equal(group_balance_mask(sg), mask));

	/*
	 * Initialize sgc->capacity such that even if we mess up the
	 * domains and no possible iteration will get us here, we won't
	 * die on a /0 trap.
	 */
	sg_span = sched_group_span(sg);
	sg->sgc->capacity = SCHED_CAPACITY_SCALE * cpumask_weight(sg_span);
	sg->sgc->min_capacity = SCHED_CAPACITY_SCALE;
	sg->sgc->max_capacity = SCHED_CAPACITY_SCALE;
}

/*
 * 非规则 NUMA 拓扑中，同层兄弟域的 child 可能超出当前域。
 * 沿其后代下探到可用范围，并跳过最终会被裁剪的等范围层级。
 */
static struct sched_domain *
find_descended_sibling(struct sched_domain *sd, struct sched_domain *sibling)
{
	/*
	 * The proper descendant would be the one whose child won't span out
	 * of sd
	 */
	while (sibling->child &&
	       !cpumask_subset(sched_domain_span(sibling->child),
			       sched_domain_span(sd)))
		sibling = sibling->child;

	/*
	 * As we are referencing sgc across different topology level, we need
	 * to go down to skip those sched_domains which don't contribute to
	 * scheduling because they will be degenerated in cpu_attach_domain
	 */
	while (sibling->child &&
	       cpumask_equal(sched_domain_span(sibling->child),
			     sched_domain_span(sibling)))
		sibling = sibling->child;

	return sibling;
}

/*
 * 构建允许相互重叠的 NUMA 组：从本 CPU 开始，按需添加子域对应的组，
 * 直到 covered 覆盖整个域。首组因而代表本 CPU 的子域，而非任意远端组。
 */
static int
build_overlap_sched_groups(struct sched_domain *sd, int cpu)
{
	struct sched_group *first = NULL, *last = NULL, *sg;
	const struct cpumask *span = sched_domain_span(sd);
	struct cpumask *covered = sched_domains_tmpmask;
	struct sd_data *sdd = sd->private;
	struct sched_domain *sibling;
	int i;

	cpumask_clear(covered);

	for_each_cpu_wrap(i, span, cpu) {
		struct cpumask *sg_span;

		if (cpumask_test_cpu(i, covered))
			continue;

		sibling = *per_cpu_ptr(sdd->sd, i);

		/*
		 * Asymmetric node setups can result in situations where the
		 * domain tree is of unequal depth, make sure to skip domains
		 * that already cover the entire range.
		 *
		 * In that case build_sched_domains() will have terminated the
		 * iteration early and our sibling sd spans will be empty.
		 * Domains should always include the CPU they're built on, so
		 * check that.
		 */
		if (!cpumask_test_cpu(i, sched_domain_span(sibling)))
			continue;

		/*
		 * Usually we build sched_group by sibling's child sched_domain
		 * But for machines whose NUMA diameter are 3 or above, we move
		 * to build sched_group by sibling's proper descendant's child
		 * domain because sibling's child sched_domain will span out of
		 * the sched_domain being built as below.
		 *
		 * Smallest diameter=3 topology is:
		 *
		 *   node   0   1   2   3
		 *     0:  10  20  30  40
		 *     1:  20  10  20  30
		 *     2:  30  20  10  20
		 *     3:  40  30  20  10
		 *
		 *   0 --- 1 --- 2 --- 3
		 *
		 * NUMA-3       0-3             N/A             N/A             0-3
		 *  groups:     {0-2},{1-3}                                     {1-3},{0-2}
		 *
		 * NUMA-2       0-2             0-3             0-3             1-3
		 *  groups:     {0-1},{1-3}     {0-2},{2-3}     {1-3},{0-1}     {2-3},{0-2}
		 *
		 * NUMA-1       0-1             0-2             1-3             2-3
		 *  groups:     {0},{1}         {1},{2},{0}     {2},{3},{1}     {3},{2}
		 *
		 * NUMA-0       0               1               2               3
		 *
		 * The NUMA-2 groups for nodes 0 and 3 are obviously buggered, as the
		 * group span isn't a subset of the domain span.
		 */
		if (sibling->child &&
		    !cpumask_subset(sched_domain_span(sibling->child), span))
			sibling = find_descended_sibling(sd, sibling);

		sg = build_group_from_child_sched_domain(sibling, cpu);
		if (!sg)
			goto fail;

		sg_span = sched_group_span(sg);
		cpumask_or(covered, covered, sg_span);

		init_overlap_sched_group(sibling, sg);

		if (!first)
			first = sg;
		if (last)
			last->next = sg;
		last = sg;
		last->next = first;
	}
	sd->groups = first;

	return 0;

fail:
	/* 回收私有组；容量对象仍由 sd_data 临时分配表统一清理。 */
	free_sched_groups(first, 0);

	return -ENOMEM;
}


/*
 * Package topology (also see the load-balance blurb in fair.c)
 *
 * The scheduler builds a tree structure to represent a number of important
 * topology features. By default (default_topology[]) these include:
 *
 *  - Simultaneous multithreading (SMT)
 *  - Multi-Core Cache (MC)
 *  - Package (PKG)
 *
 * Where the last one more or less denotes everything up to a NUMA node.
 *
 * The tree consists of 3 primary data structures:
 *
 *	sched_domain -> sched_group -> sched_group_capacity
 *	    ^ ^             ^ ^
 *          `-'             `-'
 *
 * The sched_domains are per-CPU and have a two way link (parent & child) and
 * denote the ever growing mask of CPUs belonging to that level of topology.
 *
 * Each sched_domain has a circular (double) linked list of sched_group's, each
 * denoting the domains of the level below (or individual CPUs in case of the
 * first domain level). The sched_group linked by a sched_domain includes the
 * CPU of that sched_domain [*].
 *
 * Take for instance a 2 threaded, 2 core, 2 cache cluster part:
 *
 * CPU   0   1   2   3   4   5   6   7
 *
 * PKG  [                             ]
 * MC   [             ] [             ]
 * SMT  [     ] [     ] [     ] [     ]
 *
 *  - or -
 *
 * PKG  0-7 0-7 0-7 0-7 0-7 0-7 0-7 0-7
 * MC	0-3 0-3 0-3 0-3 4-7 4-7 4-7 4-7
 * SMT  0-1 0-1 2-3 2-3 4-5 4-5 6-7 6-7
 *
 * CPU   0   1   2   3   4   5   6   7
 *
 * One way to think about it is: sched_domain moves you up and down among these
 * topology levels, while sched_group moves you sideways through it, at child
 * domain granularity.
 *
 * sched_group_capacity ensures each unique sched_group has shared storage.
 *
 * There are two related construction problems, both require a CPU that
 * uniquely identify each group (for a given domain):
 *
 *  - The first is the balance_cpu (see should_we_balance() and the
 *    load-balance blurb in fair.c); for each group we only want 1 CPU to
 *    continue balancing at a higher domain.
 *
 *  - The second is the sched_group_capacity; we want all identical groups
 *    to share a single sched_group_capacity.
 *
 * Since these topologies are exclusive by construction. That is, its
 * impossible for an SMT thread to belong to multiple cores, and cores to
 * be part of multiple caches. There is a very clear and unique location
 * for each CPU in the hierarchy.
 *
 * Therefore computing a unique CPU for each group is trivial (the iteration
 * mask is redundant and set all 1s; all CPUs in a group will end up at _that_
 * group), we can simply pick the first CPU in each group.
 *
 *
 * [*] in other words, the first group of each domain is its child domain.
 */

/*
 * 普通拓扑中，以 child span 的首个 CPU 唯一标识组；底层则每 CPU 一组。
 * 不同 CPU 的域可引用同一组和容量对象，仅首次引用时初始化内容。
 */
static struct sched_group *get_group(int cpu, struct sd_data *sdd)
{
	struct sched_domain *sd = *per_cpu_ptr(sdd->sd, cpu);
	struct sched_domain *child = sd->child;
	struct sched_group *sg;
	bool already_visited;

	if (child)
		cpu = cpumask_first(sched_domain_span(child));

	sg = *per_cpu_ptr(sdd->sg, cpu);
	sg->sgc = *per_cpu_ptr(sdd->sgc, cpu);

	/* Increase refcounts for claim_allocations: */
	already_visited = atomic_inc_return(&sg->ref) > 1;
	/* sgc visits should follow a similar trend as sg */
	WARN_ON(already_visited != (atomic_inc_return(&sg->sgc->ref) > 1));

	/* If we have already visited that group, it's already initialized. */
	if (already_visited)
		return sg;

	if (child) {
		cpumask_copy(sched_group_span(sg), sched_domain_span(child));
		cpumask_copy(group_balance_mask(sg), sched_group_span(sg));
		sg->flags = child->flags;
	} else {
		cpumask_set_cpu(cpu, sched_group_span(sg));
		cpumask_set_cpu(cpu, group_balance_mask(sg));
	}

	sg->sgc->capacity = SCHED_CAPACITY_SCALE * cpumask_weight(sched_group_span(sg));
	sg->sgc->min_capacity = SCHED_CAPACITY_SCALE;
	sg->sgc->max_capacity = SCHED_CAPACITY_SCALE;

	return sg;
}

/*
 * build_sched_groups will build a circular linked list of the groups
 * covered by the given span, will set each group's ->cpumask correctly,
 * and will initialize their ->sgc.
 *
 * Assumes the sched_domain tree is fully constructed
 *
 * covered 跳过已经归入某个子域组的 CPU；从 cpu 开始环绕遍历，保证本地组
 * 位于链表首部。分组不重叠，可复用 get_group() 返回的共享节点。
 */
static int
build_sched_groups(struct sched_domain *sd, int cpu)
{
	struct sched_group *first = NULL, *last = NULL;
	struct sd_data *sdd = sd->private;
	const struct cpumask *span = sched_domain_span(sd);
	struct cpumask *covered;
	int i;

	lockdep_assert_held(&sched_domains_mutex);
	covered = sched_domains_tmpmask;

	cpumask_clear(covered);

	for_each_cpu_wrap(i, span, cpu) {
		struct sched_group *sg;

		if (cpumask_test_cpu(i, covered))
			continue;

		sg = get_group(i, sdd);

		cpumask_or(covered, covered, sched_group_span(sg));

		if (!first)
			first = sg;
		if (last)
			last->next = sg;
		last = sg;
	}
	last->next = first;
	sd->groups = first;

	return 0;
}

/*
 * Initialize sched groups cpu_capacity.
 *
 * cpu_capacity indicates the capacity of sched group, which is used while
 * distributing the load between different sched groups in a sched domain.
 * Typically cpu_capacity for all the groups in a sched domain will be same
 * unless there are asymmetries in the topology. If there are asymmetries,
 * group having more cpu_capacity will pickup more load compared to the
 * group having less cpu_capacity.
 *
 * 同时缓存逻辑 CPU 数、去除 SMT 重复后的核心数以及非对称打包首选 CPU。
 * 共享容量只由该组的 balance CPU 更新，避免每个引用者重复计算。
 */
static void init_sched_groups_capacity(int cpu, struct sched_domain *sd)
{
	struct sched_group *sg = sd->groups;
	struct cpumask *mask = sched_domains_tmpmask2;

	WARN_ON(!sg);

	do {
		int cpu, cores = 0, max_cpu = -1;

		sg->group_weight = cpumask_weight(sched_group_span(sg));

		/* 每计入一个物理核心，就从临时集合中移除其余 SMT 线程。 */
		cpumask_copy(mask, sched_group_span(sg));
		for_each_cpu(cpu, mask) {
			cores++;
#ifdef CONFIG_SCHED_SMT
			cpumask_andnot(mask, mask, cpu_smt_mask(cpu));
#endif
		}
		sg->cores = cores;

		if (!(sd->flags & SD_ASYM_PACKING))
			goto next;

		for_each_cpu(cpu, sched_group_span(sg)) {
			if (max_cpu < 0)
				max_cpu = cpu;
			else if (sched_asym_prefer(cpu, max_cpu))
				max_cpu = cpu;
		}
		sg->asym_prefer_cpu = max_cpu;

next:
		sg = sg->next;
	} while (sg != sd->groups);

	if (cpu != group_balance_cpu(sg))
		return;

	update_group_capacity(sd, cpu);
}

/*
 * Update the "asym_prefer_cpu" when arch_asym_cpu_priority() changes.
 *
 * 沿本 CPU 的域链更新本地组：优先级提高时尝试成为首选，原首选降级时
 * 重新扫描组成员。这里的优先级是架构打包顺序，不是任务调度优先级。
 */
void sched_update_asym_prefer_cpu(int cpu, int old_prio, int new_prio)
{
	int asym_prefer_cpu = cpu;
	struct sched_domain *sd;

	guard(rcu)();

	for_each_domain(cpu, sd) {
		struct sched_group *sg;
		int group_cpu;

		if (!(sd->flags & SD_ASYM_PACKING))
			continue;

		/*
		 * Groups of overlapping domain are replicated per NUMA
		 * node and will require updating "asym_prefer_cpu" on
		 * each local copy.
		 *
		 * If you are hitting this warning, consider moving
		 * "sg->asym_prefer_cpu" to "sg->sgc->asym_prefer_cpu"
		 * which is shared by all the overlapping groups.
		 */
		WARN_ON_ONCE(sd->flags & SD_NUMA);

		sg = sd->groups;
		if (cpu != sg->asym_prefer_cpu) {
			/*
			 * Since the parent is a superset of the current group,
			 * if the cpu is not the "asym_prefer_cpu" at the
			 * current level, it cannot be the preferred CPU at a
			 * higher levels either.
			 */
			if (!sched_asym_prefer(cpu, sg->asym_prefer_cpu))
				return;

			WRITE_ONCE(sg->asym_prefer_cpu, cpu);
			continue;
		}

		/* Ranking has improved; CPU is still the preferred one. */
		if (new_prio >= old_prio)
			continue;

		for_each_cpu(group_cpu, sched_group_span(sg)) {
			if (sched_asym_prefer(group_cpu, asym_prefer_cpu))
				asym_prefer_cpu = group_cpu;
		}

		WRITE_ONCE(sg->asym_prefer_cpu, asym_prefer_cpu);
	}
}

/*
 * Set of available CPUs grouped by their corresponding capacities
 * Each list entry contains a CPU mask reflecting CPUs that share the same
 * capacity.
 * The lifespan of data is unlimited.
 */
LIST_HEAD(asym_cap_list);

/*
 * Verify whether there is any CPU capacity asymmetry in a given sched domain.
 * Provides sd_flags reflecting the asymmetry scope.
 *
 * count 统计域内容量类别数；miss 统计分区存在但本域未覆盖的类别数。
 * FULL 表示包含分区内全部容量类别，并不要求包含这些类别的所有 CPU。
 */
static inline int
asym_cpu_capacity_classify(const struct cpumask *sd_span,
			   const struct cpumask *cpu_map)
{
	struct asym_cap_data *entry;
	int count = 0, miss = 0;

	/*
	 * Count how many unique CPU capacities this domain spans across
	 * (compare sched_domain CPUs mask with ones representing  available
	 * CPUs capacities). Take into account CPUs that might be offline:
	 * skip those.
	 */
	list_for_each_entry(entry, &asym_cap_list, link) {
		if (cpumask_intersects(sd_span, cpu_capacity_span(entry)))
			++count;
		else if (cpumask_intersects(cpu_map, cpu_capacity_span(entry)))
			++miss;
	}

	WARN_ON_ONCE(!count && !list_empty(&asym_cap_list));

	/* No asymmetry detected */
	if (count < 2)
		return 0;
	/* Some of the available CPU capacity values have not been detected */
	if (miss)
		return SD_ASYM_CPUCAPACITY;

	/* Full asymmetry */
	return SD_ASYM_CPUCAPACITY | SD_ASYM_CPUCAPACITY_FULL;

}

/* 容量类别从 RCU 链表摘除后延迟释放，保护并行遍历该链表的读者。 */
static void free_asym_cap_entry(struct rcu_head *head)
{
	struct asym_cap_data *entry = container_of(head, struct asym_cap_data, rcu);
	kfree(entry);
}

/* 将 CPU 加入对应容量类别，必要时新建条目，保持容量降序排列。 */
static inline void asym_cpu_capacity_update_data(int cpu)
{
	unsigned long capacity = arch_scale_cpu_capacity(cpu);
	struct asym_cap_data *insert_entry = NULL;
	struct asym_cap_data *entry;

	/*
	 * Search if capacity already exits. If not, track which the entry
	 * where we should insert to keep the list ordered descending.
	 */
	list_for_each_entry(entry, &asym_cap_list, link) {
		if (capacity == entry->capacity)
			goto done;
		else if (!insert_entry && capacity > entry->capacity)
			insert_entry = list_prev_entry(entry, link);
	}

	entry = kzalloc(sizeof(*entry) + cpumask_size(), GFP_KERNEL);
	if (WARN_ONCE(!entry, "Failed to allocate memory for asymmetry data\n"))
		return;
	entry->capacity = capacity;

	/* If NULL then the new capacity is the smallest, add last. */
	if (!insert_entry)
		list_add_tail_rcu(&entry->link, &asym_cap_list);
	else
		list_add_rcu(&entry->link, &insert_entry->link);
done:
	__cpumask_set_cpu(cpu, cpu_capacity_span(entry));
}

/*
 * Build-up/update list of CPUs grouped by their capacities
 * An update requires explicit request to rebuild sched domains
 * with state indicating CPU topology changes.
 *
 * 扫描可参与域均衡的 possible CPU，而非仅扫描当前在线 CPU；随后按分区
 * 掩码分类。只有一种容量时清空类别链表，表示无需非对称处理。
 */
static void asym_cpu_capacity_scan(void)
{
	struct asym_cap_data *entry, *next;
	int cpu;

	list_for_each_entry(entry, &asym_cap_list, link)
		cpumask_clear(cpu_capacity_span(entry));

	for_each_cpu_and(cpu, cpu_possible_mask, housekeeping_cpumask(HK_TYPE_DOMAIN))
		asym_cpu_capacity_update_data(cpu);

	list_for_each_entry_safe(entry, next, &asym_cap_list, link) {
		if (cpumask_empty(cpu_capacity_span(entry))) {
			list_del_rcu(&entry->link);
			call_rcu(&entry->rcu, free_asym_cap_entry);
		}
	}

	/*
	 * Only one capacity value has been detected i.e. this system is symmetric.
	 * No need to keep this data around.
	 */
	if (list_is_singular(&asym_cap_list)) {
		entry = list_first_entry(&asym_cap_list, typeof(*entry), link);
		list_del_rcu(&entry->link);
		call_rcu(&entry->rcu, free_asym_cap_entry);
	}
}

/*
 * Initializers for schedule domains
 * Non-inlined to reduce accumulated stack pressure in build_sched_domains()
 */

static int default_relax_domain_level = -1;
int sched_domain_level_max;

/* 解析默认的域放宽级别，供未指定分区属性的构建路径使用。 */
static int __init setup_relax_domain_level(char *str)
{
	if (kstrtoint(str, 0, &default_relax_domain_level))
		pr_warn("Unable to set relax_domain_level\n");

	return 1;
}
__setup("relax_domain_level=", setup_relax_domain_level);

/*
 * 分区属性优先于启动参数；达到请求级别时关闭唤醒均衡和新空闲均衡。
 * 这里只调整这些触发方式，不会完全禁止周期性负载均衡。
 */
static void set_domain_attribute(struct sched_domain *sd,
				 struct sched_domain_attr *attr)
{
	int request;

	if (!attr || attr->relax_domain_level < 0) {
		if (default_relax_domain_level < 0)
			return;
		request = default_relax_domain_level;
	} else
		request = attr->relax_domain_level;

	if (sd->level >= request) {
		/* Turn off idle balance on this domain: */
		sd->flags &= ~(SD_BALANCE_WAKE|SD_BALANCE_NEWIDLE);
	}
}

static void __sdt_free(const struct cpumask *cpu_map);
static int __sdt_alloc(const struct cpumask *cpu_map);

/*
 * 按分配进度统一清理临时资源。已被运行队列引用的 root_domain 不再回收，
 * 已由 claim_allocations() 转移所有权的域/组也不在临时表中释放。
 */
static void __free_domain_allocs(struct s_data *d, enum s_alloc what,
				 const struct cpumask *cpu_map)
{
	switch (what) {
	case sa_rootdomain:
		if (!atomic_read(&d->rd->refcount))
			free_rootdomain(&d->rd->rcu);
		fallthrough;
	case sa_sd:
		free_percpu(d->sd);
		fallthrough;
	case sa_sd_storage:
		__sdt_free(cpu_map);
		fallthrough;
	case sa_none:
		break;
	}
}

/*
 * 分三步准备每层拓扑存储、每 CPU 底层域指针和根域。
 * 返回值是调用者应执行的清理层级，只有 sa_rootdomain 表示全部分配成功。
 */
static enum s_alloc
__visit_domain_allocation_hell(struct s_data *d, const struct cpumask *cpu_map)
{
	memset(d, 0, sizeof(*d));

	if (__sdt_alloc(cpu_map))
		return sa_sd_storage;
	d->sd = alloc_percpu(struct sched_domain *);
	if (!d->sd)
		return sa_sd_storage;
	d->rd = alloc_rootdomain();
	if (!d->rd)
		return sa_sd;

	return sa_rootdomain;
}

/*
 * NULL the sd_data elements we've used to build the sched_domain and
 * sched_group structure so that the subsequent __free_domain_allocs()
 * will not free the data we're using.
 *
 * sd_data 初始拥有全部预分配对象；共享对象及当前域被接管后，
 * 就清空对应槽位。未被用到的对象仍留在表中，构建结束时统一释放。
 */
static void claim_allocations(int cpu, struct sched_domain *sd)
{
	struct sd_data *sdd = sd->private;

	WARN_ON_ONCE(*per_cpu_ptr(sdd->sd, cpu) != sd);
	*per_cpu_ptr(sdd->sd, cpu) = NULL;

	if (atomic_read(&(*per_cpu_ptr(sdd->sds, cpu))->ref))
		*per_cpu_ptr(sdd->sds, cpu) = NULL;

	if (atomic_read(&(*per_cpu_ptr(sdd->sg, cpu))->ref))
		*per_cpu_ptr(sdd->sg, cpu) = NULL;

	if (atomic_read(&(*per_cpu_ptr(sdd->sgc, cpu))->ref))
		*per_cpu_ptr(sdd->sgc, cpu) = NULL;
}

#ifdef CONFIG_NUMA
enum numa_topology_type sched_numa_topology_type;

static int			sched_domains_numa_levels;

int				sched_max_numa_distance;
static int			*sched_domains_numa_distance;
/* [距离层级][起点节点]：距离不超过该层阈值的节点 CPU 并集。 */
static struct cpumask		***sched_domains_numa_masks;
#endif /* CONFIG_NUMA */

/*
 * SD_flags allowed in topology descriptions.
 *
 * These flags are purely descriptive of the topology and do not prescribe
 * behaviour. Behaviour is artificial and mapped in the below sd_init()
 * function. For details, see include/linux/sched/sd_flags.h.
 *
 *   SD_SHARE_CPUCAPACITY
 *   SD_SHARE_LLC
 *   SD_CLUSTER
 *   SD_NUMA
 *
 * Odd one out, which beside describing the topology has a quirk also
 * prescribes the desired behaviour that goes along with it:
 *
 *   SD_ASYM_PACKING        - describes SMT quirks
 */
#define TOPOLOGY_SD_FLAGS		\
	(SD_SHARE_CPUCAPACITY	|	\
	 SD_CLUSTER		|	\
	 SD_SHARE_LLC		|	\
	 SD_NUMA		|	\
	 SD_ASYM_PACKING)

/*
 * 初始化一个 CPU 在某拓扑层上的域：先将架构掩码限制在当前分区内，
 * 再把硬件共享关系和容量差异映射为负载均衡行为及参数。
 * 返回的域仍处于构建阶段，父链、组链及最终容量将在后续阶段建立。
 */
static struct sched_domain *
sd_init(struct sched_domain_topology_level *tl,
	const struct cpumask *cpu_map,
	struct sched_domain *child, int cpu)
{
	struct sd_data *sdd = &tl->data;
	struct sched_domain *sd = *per_cpu_ptr(sdd->sd, cpu);
	int sd_id, sd_weight, sd_flags = 0;
	struct cpumask *sd_span;

	sd_span = sched_domain_span(sd);
	cpumask_and(sd_span, cpu_map, tl->mask(tl, cpu));
	sd_weight = cpumask_weight(sd_span);
	sd_id = cpumask_first(sd_span);

	if (tl->sd_flags)
		sd_flags = (*tl->sd_flags)();
	if (WARN_ONCE(sd_flags & ~TOPOLOGY_SD_FLAGS,
		      "wrong sd_flags in topology description\n"))
		sd_flags &= TOPOLOGY_SD_FLAGS;
	/* 从分区容量类别推导非对称性，不由拓扑回调直接指定。 */
	sd_flags |= asym_cpu_capacity_classify(sd_span, cpu_map);

	*sd = (struct sched_domain){
		.min_interval		= sd_weight,
		.max_interval		= 2*sd_weight,
		.busy_factor		= 16,
		.imbalance_pct		= 117,

		.cache_nice_tries	= 0,

		.flags			= 1*SD_BALANCE_NEWIDLE
					| 1*SD_BALANCE_EXEC
					| 1*SD_BALANCE_FORK
					| 0*SD_BALANCE_WAKE
					| 1*SD_WAKE_AFFINE
					| 0*SD_SHARE_CPUCAPACITY
					| 0*SD_SHARE_LLC
					| 0*SD_SERIALIZE
					| 1*SD_PREFER_SIBLING
					| 0*SD_NUMA
					| sd_flags
					,

		.last_balance		= jiffies,
		.balance_interval	= sd_weight,

		/* 50% success rate */
		.newidle_call		= 512,
		.newidle_success	= 256,
		.newidle_ratio		= 512,

		.max_newidle_lb_cost	= 0,
		.last_decay_max_lb_cost	= jiffies,
		.child			= child,
		.name			= tl->name,
	};

	WARN_ONCE((sd->flags & (SD_SHARE_CPUCAPACITY | SD_ASYM_CPUCAPACITY)) ==
		  (SD_SHARE_CPUCAPACITY | SD_ASYM_CPUCAPACITY),
		  "CPU capacity asymmetry not supported on SMT\n");

	/*
	 * Convert topological properties into behaviour.
	 */
	/* Don't attempt to spread across CPUs of different capacities. */
	if ((sd->flags & SD_ASYM_CPUCAPACITY) && sd->child)
		sd->child->flags &= ~SD_PREFER_SIBLING;

	if (sd->flags & SD_SHARE_CPUCAPACITY) {
		sd->imbalance_pct = 110;

	} else if (sd->flags & SD_SHARE_LLC) {
		sd->imbalance_pct = 117;
		sd->cache_nice_tries = 1;

#ifdef CONFIG_NUMA
	} else if (sd->flags & SD_NUMA) {
		sd->cache_nice_tries = 2;

		sd->flags &= ~SD_PREFER_SIBLING;
		sd->flags |= SD_SERIALIZE;
		/* 远端 NUMA 层关闭这些放置策略，避免昂贵的跨节点移动。 */
		if (sched_domains_numa_distance[tl->numa_level] > node_reclaim_distance) {
			sd->flags &= ~(SD_BALANCE_EXEC |
				       SD_BALANCE_FORK |
				       SD_WAKE_AFFINE);
		}

#endif /* CONFIG_NUMA */
	} else {
		sd->cache_nice_tries = 1;
	}

	/*
	 * For all levels sharing cache; connect a sched_domain_shared
	 * instance.
	 */
	if (sd->flags & SD_SHARE_LLC) {
		sd->shared = *per_cpu_ptr(sdd->sds, sd_id);
		atomic_inc(&sd->shared->ref);
		atomic_set(&sd->shared->nr_busy_cpus, sd_weight);
	}

	sd->private = sdd;

	return sd;
}

/* 以下回调只描述各拓扑层的共享关系和 CPU 集合，策略由 sd_init() 补全。 */
#ifdef CONFIG_SCHED_SMT
int cpu_smt_flags(void)
{
	return SD_SHARE_CPUCAPACITY | SD_SHARE_LLC;
}

const struct cpumask *tl_smt_mask(struct sched_domain_topology_level *tl, int cpu)
{
	return cpu_smt_mask(cpu);
}
#endif

#ifdef CONFIG_SCHED_CLUSTER
int cpu_cluster_flags(void)
{
	return SD_CLUSTER | SD_SHARE_LLC;
}

const struct cpumask *tl_cls_mask(struct sched_domain_topology_level *tl, int cpu)
{
	return cpu_clustergroup_mask(cpu);
}
#endif

#ifdef CONFIG_SCHED_MC
int cpu_core_flags(void)
{
	return SD_SHARE_LLC;
}

const struct cpumask *tl_mc_mask(struct sched_domain_topology_level *tl, int cpu)
{
	return cpu_coregroup_mask(cpu);
}
#endif

const struct cpumask *tl_pkg_mask(struct sched_domain_topology_level *tl, int cpu)
{
	return cpu_node_mask(cpu);
}

/*
 * Topology list, bottom-up.
 *
 * 从 SMT、簇、共享缓存到包/节点逐层扩大；可选层由配置决定。
 * 架构可替换整张表，NUMA 初始化再追加 NODE 和远端 NUMA 层。
 */
static struct sched_domain_topology_level default_topology[] = {
#ifdef CONFIG_SCHED_SMT
	SDTL_INIT(tl_smt_mask, cpu_smt_flags, SMT),
#endif

#ifdef CONFIG_SCHED_CLUSTER
	SDTL_INIT(tl_cls_mask, cpu_cluster_flags, CLS),
#endif

#ifdef CONFIG_SCHED_MC
	SDTL_INIT(tl_mc_mask, cpu_core_flags, MC),
#endif
	SDTL_INIT(tl_pkg_mask, NULL, PKG),
	{ NULL, },
};

static struct sched_domain_topology_level *sched_domain_topology =
	default_topology;
static struct sched_domain_topology_level *sched_domain_topology_saved;

#define for_each_sd_topology(tl)			\
	for (tl = sched_domain_topology; tl->mask; tl++)

/* 仅允许在 SMP 调度初始化完成前替换基础拓扑，不用于运行时换域。 */
void __init set_sched_topology(struct sched_domain_topology_level *tl)
{
	if (WARN_ON_ONCE(sched_smp_initialized))
		return;

	sched_domain_topology = tl;
	sched_domain_topology_saved = NULL;
}

#ifdef CONFIG_NUMA
static int cpu_numa_flags(void)
{
	return SD_NUMA;
}

static const struct cpumask *sd_numa_mask(struct sched_domain_topology_level *tl, int cpu)
{
	return sched_domains_numa_masks[tl->numa_level][cpu_to_node(cpu)];
}

/* 节点距离异常只打印一次并附距离矩阵，避免多层构建反复输出。 */
static void sched_numa_warn(const char *str)
{
	static int done = false;
	int i,j;

	if (done)
		return;

	done = true;

	printk(KERN_WARNING "ERROR: %s\n\n", str);

	for (i = 0; i < nr_node_ids; i++) {
		printk(KERN_WARNING "  ");
		for (j = 0; j < nr_node_ids; j++) {
			if (!node_state(i, N_CPU) || !node_state(j, N_CPU))
				printk(KERN_CONT "(%02d) ", node_distance(i,j));
			else
				printk(KERN_CONT " %02d  ", node_distance(i,j));
		}
		printk(KERN_CONT "\n");
	}
	printk(KERN_WARNING "\n");
}

/* 检查某距离值是否属于当前 NUMA 距离表；本地距离始终视为有效。 */
bool find_numa_distance(int distance)
{
	bool found = false;
	int i, *distances;

	if (distance == node_distance(0, 0))
		return true;

	rcu_read_lock();
	distances = rcu_dereference(sched_domains_numa_distance);
	if (!distances)
		goto unlock;
	for (i = 0; i < sched_domains_numa_levels; i++) {
		if (distances[i] == distance) {
			found = true;
			break;
		}
	}
unlock:
	rcu_read_unlock();

	return found;
}

#define for_each_cpu_node_but(n, nbut)		\
	for_each_node_state(n, N_CPU)		\
		if (n == nbut)			\
			continue;		\
		else

/*
 * A system can have three types of NUMA topology:
 * NUMA_DIRECT: all nodes are directly connected, or not a NUMA system
 * NUMA_GLUELESS_MESH: some nodes reachable through intermediary nodes
 * NUMA_BACKPLANE: nodes can reach other nodes through a backplane
 *
 * The difference between a glueless mesh topology and a backplane
 * topology lies in whether communication between not directly
 * connected nodes goes through intermediary nodes (where programs
 * could run), or through backplane controllers. This affects
 * placement of programs.
 *
 * The type of topology can be discerned with the following tests:
 * - If the maximum distance between any nodes is 1 hop, the system
 *   is directly connected.
 * - If for two nodes A and B, located N > 1 hops away from each other,
 *   there is an intermediary node C, which is < N hops away from both
 *   nodes A and B, the system is a glueless mesh.
 *
 * 根据最远节点对是否存在更近的中间节点区分网格与背板；offline_node 用于
 * 排除即将失去最后一个 CPU 的节点，不把它计入新的调度拓扑。
 */
static void init_numa_topology_type(int offline_node)
{
	int a, b, c, n;

	n = sched_max_numa_distance;

	if (sched_domains_numa_levels <= 2) {
		sched_numa_topology_type = NUMA_DIRECT;
		return;
	}

	for_each_cpu_node_but(a, offline_node) {
		for_each_cpu_node_but(b, offline_node) {
			/* Find two nodes furthest removed from each other. */
			if (node_distance(a, b) < n)
				continue;

			/* Is there an intermediary node between a and b? */
			for_each_cpu_node_but(c, offline_node) {
				if (node_distance(a, c) < n &&
				    node_distance(b, c) < n) {
					sched_numa_topology_type =
							NUMA_GLUELESS_MESH;
					return;
				}
			}

			sched_numa_topology_type = NUMA_BACKPLANE;
			return;
		}
	}

	pr_err("Failed to find a NUMA topology type, defaulting to DIRECT\n");
	sched_numa_topology_type = NUMA_DIRECT;
}


#define NR_DISTANCE_VALUES (1 << DISTANCE_BITS)

/*
 * 从有效节点的距离矩阵建立 NUMA 层级，并追加到架构的基础拓扑之后。
 * 距离去重排序后，每个阈值对应一层掩码，包含不超过该距离的 CPU。
 * 因而这里的 hop 是距离档位，不一定等同于硬件互连上的实际跳数。
 * offline_node 为 NUMA_NO_NODE 时不额外排除节点。
 */
void sched_init_numa(int offline_node)
{
	struct sched_domain_topology_level *tl;
	unsigned long *distance_map;
	int nr_levels = 0;
	int i, j;
	int *distances;
	struct cpumask ***masks;

	/*
	 * O(nr_nodes^2) de-duplicating selection sort -- in order to find the
	 * unique distances in the node_distance() table.
	 */
	distance_map = bitmap_alloc(NR_DISTANCE_VALUES, GFP_KERNEL);
	if (!distance_map)
		return;

	bitmap_zero(distance_map, NR_DISTANCE_VALUES);
	for_each_cpu_node_but(i, offline_node) {
		for_each_cpu_node_but(j, offline_node) {
			int distance = node_distance(i, j);

			if (distance < LOCAL_DISTANCE || distance >= NR_DISTANCE_VALUES) {
				sched_numa_warn("Invalid distance value range");
				bitmap_free(distance_map);
				return;
			}

			bitmap_set(distance_map, distance, 1);
		}
	}
	/*
	 * We can now figure out how many unique distance values there are and
	 * allocate memory accordingly.
	 */
	nr_levels = bitmap_weight(distance_map, NR_DISTANCE_VALUES);

	distances = kcalloc(nr_levels, sizeof(int), GFP_KERNEL);
	if (!distances) {
		bitmap_free(distance_map);
		return;
	}

	/* 按位图位号递增提取距离，后续层的掩码包含前面各层的范围。 */
	for (i = 0, j = 0; i < nr_levels; i++, j++) {
		j = find_next_bit(distance_map, NR_DISTANCE_VALUES, j);
		distances[i] = j;
	}
	rcu_assign_pointer(sched_domains_numa_distance, distances);

	bitmap_free(distance_map);

	/*
	 * 'nr_levels' contains the number of unique distances
	 *
	 * The sched_domains_numa_distance[] array includes the actual distance
	 * numbers.
	 */

	/*
	 * Here, we should temporarily reset sched_domains_numa_levels to 0.
	 * If it fails to allocate memory for array sched_domains_numa_masks[][],
	 * the array will contain less then 'nr_levels' members. This could be
	 * dangerous when we use it to iterate array sched_domains_numa_masks[][]
	 * in other functions.
	 *
	 * We reset it to 'nr_levels' at the end of this function.
	 */
	sched_domains_numa_levels = 0;

	masks = kzalloc(sizeof(void *) * nr_levels, GFP_KERNEL);
	if (!masks)
		return;

	/*
	 * Now for each level, construct a mask per node which contains all
	 * CPUs of nodes that are that many hops away from us.
	 */
	for (i = 0; i < nr_levels; i++) {
		masks[i] = kzalloc(nr_node_ids * sizeof(void *), GFP_KERNEL);
		if (!masks[i])
			return;

		for_each_cpu_node_but(j, offline_node) {
			struct cpumask *mask = kzalloc(cpumask_size(), GFP_KERNEL);
			int k;

			if (!mask)
				return;

			masks[i][j] = mask;

			for_each_cpu_node_but(k, offline_node) {
				if (sched_debug() && (node_distance(j, k) != node_distance(k, j)))
					sched_numa_warn("Node-distance not symmetric");

				if (node_distance(j, k) > sched_domains_numa_distance[i])
					continue;

				cpumask_or(mask, mask, cpumask_of_node(k));
			}
		}
	}
	rcu_assign_pointer(sched_domains_numa_masks, masks);

	/* Compute default topology size */
	for (i = 0; sched_domain_topology[i].mask; i++);

	tl = kzalloc((i + nr_levels + 1) *
			sizeof(struct sched_domain_topology_level), GFP_KERNEL);
	if (!tl)
		return;

	/*
	 * Copy the default topology bits..
	 */
	for (i = 0; sched_domain_topology[i].mask; i++)
		tl[i] = sched_domain_topology[i];

	/*
	 * Add the NUMA identity distance, aka single NODE.
	 */
	tl[i++] = SDTL_INIT(sd_numa_mask, NULL, NODE);

	/*
	 * .. and append 'j' levels of NUMA goodness.
	 */
	for (j = 1; j < nr_levels; i++, j++) {
		tl[i] = SDTL_INIT(sd_numa_mask, cpu_numa_flags, NUMA);
		tl[i].numa_level = j;
	}

	sched_domain_topology_saved = sched_domain_topology;
	sched_domain_topology = tl;

	sched_domains_numa_levels = nr_levels;
	WRITE_ONCE(sched_max_numa_distance, sched_domains_numa_distance[nr_levels - 1]);

	init_numa_topology_type(offline_node);
}


/*
 * 撤下已发布的 NUMA 距离表和掩码，等待 RCU 读者退出后回收，
 * 并恢复未追加 NUMA 层的基础拓扑，供随后重新初始化使用。
 */
static void sched_reset_numa(void)
{
	int nr_levels, *distances;
	struct cpumask ***masks;

	nr_levels = sched_domains_numa_levels;
	sched_domains_numa_levels = 0;
	sched_max_numa_distance = 0;
	sched_numa_topology_type = NUMA_DIRECT;
	distances = sched_domains_numa_distance;
	rcu_assign_pointer(sched_domains_numa_distance, NULL);
	masks = sched_domains_numa_masks;
	rcu_assign_pointer(sched_domains_numa_masks, NULL);
	if (distances || masks) {
		int i, j;

		synchronize_rcu();
		kfree(distances);
		for (i = 0; i < nr_levels && masks; i++) {
			if (!masks[i])
				continue;
			for_each_node(j)
				kfree(masks[i][j]);
			kfree(masks[i]);
		}
		kfree(masks);
	}
	if (sched_domain_topology_saved) {
		kfree(sched_domain_topology);
		sched_domain_topology = sched_domain_topology_saved;
		sched_domain_topology_saved = NULL;
	}
}

/*
 * Call with hotplug lock held
 *
 * 只有节点上第一个 CPU 上线或最后一个 CPU 下线才改变节点级拓扑；
 * 节点仍有其他 CPU 时，只需由掩码更新路径调整 CPU 成员。
 */
void sched_update_numa(int cpu, bool online)
{
	int node;

	node = cpu_to_node(cpu);
	/*
	 * Scheduler NUMA topology is updated when the first CPU of a
	 * node is onlined or the last CPU of a node is offlined.
	 */
	if (cpumask_weight(cpumask_of_node(node)) != 1)
		return;

	sched_reset_numa();
	sched_init_numa(online ? NUMA_NO_NODE : node);
}

/* CPU 上线时，将它加入各起点节点在相应距离阈值内的累积掩码。 */
void sched_domains_numa_masks_set(unsigned int cpu)
{
	int node = cpu_to_node(cpu);
	int i, j;

	for (i = 0; i < sched_domains_numa_levels; i++) {
		for (j = 0; j < nr_node_ids; j++) {
			if (!node_state(j, N_CPU))
				continue;

			/* Set ourselves in the remote node's masks */
			if (node_distance(j, node) <= sched_domains_numa_distance[i])
				cpumask_set_cpu(cpu, sched_domains_numa_masks[i][j]);
		}
	}
}

/* CPU 下线时从所有已分配的 NUMA 掩码中删除它，不改变距离档位本身。 */
void sched_domains_numa_masks_clear(unsigned int cpu)
{
	int i, j;

	for (i = 0; i < sched_domains_numa_levels; i++) {
		for (j = 0; j < nr_node_ids; j++) {
			if (sched_domains_numa_masks[i][j])
				cpumask_clear_cpu(cpu, sched_domains_numa_masks[i][j]);
		}
	}
}

/*
 * sched_numa_find_closest() - given the NUMA topology, find the cpu
 *                             closest to @cpu from @cpumask.
 * cpumask: cpumask to find a cpu from
 * cpu: cpu to be close to
 *
 * returns: cpu, or nr_cpu_ids when nothing found.
 *
 * 从本地距离向远端逐层搜索，在首个非空交集中分散选择 CPU。
 * 候选资格由 cpus 决定，不保证返回最小编号，也不评估负载或容量。
 */
int sched_numa_find_closest(const struct cpumask *cpus, int cpu)
{
	int i, j = cpu_to_node(cpu), found = nr_cpu_ids;
	struct cpumask ***masks;

	rcu_read_lock();
	masks = rcu_dereference(sched_domains_numa_masks);
	if (!masks)
		goto unlock;
	for (i = 0; i < sched_domains_numa_levels; i++) {
		if (!masks[i][j])
			break;
		cpu = cpumask_any_and_distribute(cpus, masks[i][j]);
		if (cpu < nr_cpu_ids) {
			found = cpu;
			break;
		}
	}
unlock:
	rcu_read_unlock();

	return found;
}

/* 二分查找状态：cpu 是从 0 开始的目标序号，w 是更近各层的候选总数。 */
struct __cmp_key {
	const struct cpumask *cpus;
	struct cpumask ***masks;
	int node;
	int cpu;
	int w;
};

/*
 * 利用累积掩码的候选数单调性定位距离层：目标序号必须满足
 * 前一层候选数 <= cpu < 当前层候选数，并记录前一层数量以换算层内序号。
 */
static int hop_cmp(const void *a, const void *b)
{
	struct cpumask **prev_hop, **cur_hop = *(struct cpumask ***)b;
	struct __cmp_key *k = (struct __cmp_key *)a;

	if (cpumask_weight_and(k->cpus, cur_hop[k->node]) <= k->cpu)
		return 1;

	if (b == k->masks) {
		k->w = 0;
		return 0;
	}

	prev_hop = *((struct cpumask ***)b - 1);
	k->w = cpumask_weight_and(k->cpus, prev_hop[k->node]);
	if (k->w <= k->cpu)
		return 0;

	return -1;
}

/**
 * sched_numa_find_nth_cpu() - given the NUMA topology, find the Nth closest CPU
 *                             from @cpus to @cpu, taking into account distance
 *                             from a given @node.
 * @cpus: cpumask to find a cpu from
 * @cpu: CPU to start searching
 * @node: NUMA node to order CPUs by distance
 *
 * 此处 cpu 实际作为从 0 开始的候选序号使用，并非起始 CPU 编号。
 * 先按到 node 的距离分层，同一距离层内再按 CPU 编号排序。
 *
 * Return: cpu, or nr_cpu_ids when nothing found.
 */
int sched_numa_find_nth_cpu(const struct cpumask *cpus, int cpu, int node)
{
	struct __cmp_key k = { .cpus = cpus, .cpu = cpu };
	struct cpumask ***hop_masks;
	int hop, ret = nr_cpu_ids;

	/* 无节点约束时，直接按在线候选的 CPU 编号取第 N 个。 */
	if (node == NUMA_NO_NODE)
		return cpumask_nth_and(cpu, cpus, cpu_online_mask);

	rcu_read_lock();

	/* CPU-less node entries are uninitialized in sched_domains_numa_masks */
	node = numa_nearest_node(node, N_CPU);
	k.node = node;

	k.masks = rcu_dereference(sched_domains_numa_masks);
	if (!k.masks)
		goto unlock;

	hop_masks = bsearch(&k, k.masks, sched_domains_numa_levels, sizeof(k.masks[0]), hop_cmp);
	if (!hop_masks)
		goto unlock;
	hop = hop_masks	- k.masks;

	/* 累积掩码减去前一层后，才是本距离档位新增的候选 CPU。 */
	ret = hop ?
		cpumask_nth_and_andnot(cpu - k.w, cpus, k.masks[hop][node], k.masks[hop-1][node]) :
		cpumask_nth_and(cpu, cpus, k.masks[0][node]);
unlock:
	rcu_read_unlock();
	return ret;
}
EXPORT_SYMBOL_GPL(sched_numa_find_nth_cpu);

/**
 * sched_numa_hop_mask() - Get the cpumask of CPUs at most @hops hops away from
 *                         @node
 * @node: The node to count hops from.
 * @hops: Include CPUs up to that many hops away. 0 means local node.
 *
 * Return: On success, a pointer to a cpumask of CPUs at most @hops away from
 * @node, an error value otherwise.
 *
 * Requires rcu_lock to be held. Returned cpumask is only valid within that
 * read-side section, copy it if required beyond that.
 *
 * Note that not all hops are equal in distance; see sched_init_numa() for how
 * distances and masks are handled.
 * Also note that this is a reflection of sched_domains_numa_masks, which may change
 * during the lifetime of the system (offline nodes are taken out of the masks).
 *
 * 返回内部掩码而非副本，调用者须持有 RCU 读锁且不得修改它。
 * hops 是距离层索引，返回范围包含更近层，并非仅包含恰好该距离的 CPU。
 */
const struct cpumask *sched_numa_hop_mask(unsigned int node, unsigned int hops)
{
	struct cpumask ***masks;

	if (node >= nr_node_ids || hops >= sched_domains_numa_levels)
		return ERR_PTR(-EINVAL);

	masks = rcu_dereference(sched_domains_numa_masks);
	if (!masks)
		return ERR_PTR(-EBUSY);

	return masks[hops][node];
}
EXPORT_SYMBOL_GPL(sched_numa_hop_mask);

#endif /* CONFIG_NUMA */

/*
 * 为每个拓扑层、每个分区 CPU 预分配域、LLC 共享状态、组及容量对象。
 * 此时按 CPU 分配只是提供候选存储，真正共享哪些对象由后续构建决定；
 * 任一步失败都交给 __sdt_free() 清理已经记录在 sd_data 中的部分分配。
 */
static int __sdt_alloc(const struct cpumask *cpu_map)
{
	struct sched_domain_topology_level *tl;
	int j;

	for_each_sd_topology(tl) {
		struct sd_data *sdd = &tl->data;

		sdd->sd = alloc_percpu(struct sched_domain *);
		if (!sdd->sd)
			return -ENOMEM;

		sdd->sds = alloc_percpu(struct sched_domain_shared *);
		if (!sdd->sds)
			return -ENOMEM;

		sdd->sg = alloc_percpu(struct sched_group *);
		if (!sdd->sg)
			return -ENOMEM;

		sdd->sgc = alloc_percpu(struct sched_group_capacity *);
		if (!sdd->sgc)
			return -ENOMEM;

		for_each_cpu(j, cpu_map) {
			struct sched_domain *sd;
			struct sched_domain_shared *sds;
			struct sched_group *sg;
			struct sched_group_capacity *sgc;

			sd = kzalloc_node(sizeof(struct sched_domain) + cpumask_size(),
					GFP_KERNEL, cpu_to_node(j));
			if (!sd)
				return -ENOMEM;

			*per_cpu_ptr(sdd->sd, j) = sd;

			sds = kzalloc_node(sizeof(struct sched_domain_shared),
					GFP_KERNEL, cpu_to_node(j));
			if (!sds)
				return -ENOMEM;

			*per_cpu_ptr(sdd->sds, j) = sds;

			sg = kzalloc_node(sizeof(struct sched_group) + cpumask_size(),
					GFP_KERNEL, cpu_to_node(j));
			if (!sg)
				return -ENOMEM;

			sg->next = sg;

			*per_cpu_ptr(sdd->sg, j) = sg;

			sgc = kzalloc_node(sizeof(struct sched_group_capacity) + cpumask_size(),
					GFP_KERNEL, cpu_to_node(j));
			if (!sgc)
				return -ENOMEM;

			sgc->id = j;

			*per_cpu_ptr(sdd->sgc, j) = sgc;
		}
	}

	return 0;
}

/*
 * 回收临时表中剩余对象，并清空各层的 per-CPU 指针表。
 * 成功构建已接管的对象槽位为 NULL；失败构建的 NUMA 私有组需额外回收，
 * 其容量对象仍由 sdd->sgc 表管理，不能随组重复释放。
 */
static void __sdt_free(const struct cpumask *cpu_map)
{
	struct sched_domain_topology_level *tl;
	int j;

	for_each_sd_topology(tl) {
		struct sd_data *sdd = &tl->data;

		for_each_cpu(j, cpu_map) {
			struct sched_domain *sd;

			if (sdd->sd) {
				sd = *per_cpu_ptr(sdd->sd, j);
				if (sd && (sd->flags & SD_NUMA))
					free_sched_groups(sd->groups, 0);
				kfree(*per_cpu_ptr(sdd->sd, j));
			}

			if (sdd->sds)
				kfree(*per_cpu_ptr(sdd->sds, j));
			if (sdd->sg)
				kfree(*per_cpu_ptr(sdd->sg, j));
			if (sdd->sgc)
				kfree(*per_cpu_ptr(sdd->sgc, j));
		}
		free_percpu(sdd->sd);
		sdd->sd = NULL;
		free_percpu(sdd->sds);
		sdd->sds = NULL;
		free_percpu(sdd->sg);
		sdd->sg = NULL;
		free_percpu(sdd->sgc);
		sdd->sgc = NULL;
	}
}

/* 初始化单层域并接上 child；强制父域至少覆盖子域，再应用分区属性。 */
static struct sched_domain *build_sched_domain(struct sched_domain_topology_level *tl,
		const struct cpumask *cpu_map, struct sched_domain_attr *attr,
		struct sched_domain *child, int cpu)
{
	struct sched_domain *sd = sd_init(tl, cpu_map, child, cpu);

	if (child) {
		sd->level = child->level + 1;
		sched_domain_level_max = max(sched_domain_level_max, sd->level);
		child->parent = sd;

		if (!cpumask_subset(sched_domain_span(child),
				    sched_domain_span(sd))) {
			pr_err("BUG: arch topology borken\n");
			pr_err("     the %s domain not a subset of the %s domain\n",
					child->name, sd->name);
			/* Fixup, ensure @sd has at least @child CPUs. */
			cpumask_or(sched_domain_span(sd),
				   sched_domain_span(sd),
				   sched_domain_span(child));
		}

	}
	set_domain_attribute(sd, attr);

	return sd;
}

/*
 * Ensure topology masks are sane, i.e. there are no conflicts (overlaps) for
 * any two given CPUs on non-NUMA topology levels.
 *
 * 同一普通拓扑层的任意两个掩码必须完全相同或完全不相交，否则按首 CPU
 * 共享 sched_group 时会改坏其他域的循环链表；NUMA 层有独立组，不受此限。
 */
static bool topology_span_sane(const struct cpumask *cpu_map)
{
	struct sched_domain_topology_level *tl;
	struct cpumask *covered, *id_seen;
	int cpu;

	lockdep_assert_held(&sched_domains_mutex);
	covered = sched_domains_tmpmask;
	id_seen = sched_domains_tmpmask2;

	for_each_sd_topology(tl) {
		int tl_common_flags = 0;

		if (tl->sd_flags)
			tl_common_flags = (*tl->sd_flags)();

		/* NUMA levels are allowed to overlap */
		if (tl_common_flags & SD_NUMA)
			continue;

		cpumask_clear(covered);
		cpumask_clear(id_seen);

		/*
		 * Non-NUMA levels cannot partially overlap - they must be either
		 * completely equal or completely disjoint. Otherwise we can end up
		 * breaking the sched_group lists - i.e. a later get_group() pass
		 * breaks the linking done for an earlier span.
		 */
		for_each_cpu(cpu, cpu_map) {
			const struct cpumask *tl_cpu_mask = tl->mask(tl, cpu);
			int id;

			/* lowest bit set in this mask is used as a unique id */
			id = cpumask_first(tl_cpu_mask);

			if (cpumask_test_cpu(id, id_seen)) {
				/* First CPU has already been seen, ensure identical spans */
				if (!cpumask_equal(tl->mask(tl, id), tl_cpu_mask))
					return false;
			} else {
				/* First CPU hasn't been seen before, ensure it's a completely new span */
				if (cpumask_intersects(tl_cpu_mask, covered))
					return false;

				cpumask_or(covered, covered, tl_cpu_mask);
				cpumask_set_cpu(id, id_seen);
			}
		}
	}
	return true;
}

/*
 * Build sched domains for a given set of CPUs and attach the sched domains
 * to the individual CPUs
 *
 * 分阶段构建：纵向域链 -> 横向组链 -> NUMA 容忍度 -> 接管并初始化容量
 * -> 裁剪并挂接。失败时由临时表回滚，挂接后的生命周期由运行队列、
 * 共享引用计数和 RCU 接管，避免向读者发布半成品。
 */
static int
build_sched_domains(const struct cpumask *cpu_map, struct sched_domain_attr *attr)
{
	enum s_alloc alloc_state = sa_none;
	struct sched_domain *sd;
	struct s_data d;
	struct rq *rq = NULL;
	int i, ret = -ENOMEM;
	bool has_asym = false;
	bool has_cluster = false;

	if (WARN_ON(cpumask_empty(cpu_map)))
		goto error;

	alloc_state = __visit_domain_allocation_hell(&d, cpu_map);
	if (alloc_state != sa_rootdomain)
		goto error;

	/* Set up domains for CPUs specified by the cpu_map: */
	for_each_cpu(i, cpu_map) {
		struct sched_domain_topology_level *tl;

		sd = NULL;
		for_each_sd_topology(tl) {

			sd = build_sched_domain(tl, cpu_map, attr, sd, i);

			has_asym |= sd->flags & SD_ASYM_CPUCAPACITY;

			if (tl == sched_domain_topology)
				*per_cpu_ptr(d.sd, i) = sd;
			/* 已覆盖整个分区，无需继续创建更高层的重复域。 */
			if (cpumask_equal(cpu_map, sched_domain_span(sd)))
				break;
		}
	}

	if (WARN_ON(!topology_span_sane(cpu_map)))
		goto error;

	/* 全部域链就绪后，组构建才能引用其他 CPU 的同层或子层域。 */
	/* Build the groups for the domains */
	for_each_cpu(i, cpu_map) {
		for (sd = *per_cpu_ptr(d.sd, i); sd; sd = sd->parent) {
			sd->span_weight = cpumask_weight(sched_domain_span(sd));
			if (sd->flags & SD_NUMA) {
				if (build_overlap_sched_groups(sd, i))
					goto error;
			} else {
				if (build_sched_groups(sd, i))
					goto error;
			}
		}
	}

	/*
	 * Calculate an allowed NUMA imbalance such that LLCs do not get
	 * imbalanced.
	 */
	for_each_cpu(i, cpu_map) {
		unsigned int imb = 0;
		unsigned int imb_span = 1;

		for (sd = *per_cpu_ptr(d.sd, i); sd; sd = sd->parent) {
			struct sched_domain *child = sd->child;

			if (!(sd->flags & SD_SHARE_LLC) && child &&
			    (child->flags & SD_SHARE_LLC)) {
				struct sched_domain __rcu *top_p;
				unsigned int nr_llcs;

				/*
				 * For a single LLC per node, allow an
				 * imbalance up to 12.5% of the node. This is
				 * arbitrary cutoff based two factors -- SMT and
				 * memory channels. For SMT-2, the intent is to
				 * avoid premature sharing of HT resources but
				 * SMT-4 or SMT-8 *may* benefit from a different
				 * cutoff. For memory channels, this is a very
				 * rough estimate of how many channels may be
				 * active and is based on recent CPUs with
				 * many cores.
				 *
				 * For multiple LLCs, allow an imbalance
				 * until multiple tasks would share an LLC
				 * on one node while LLCs on another node
				 * remain idle. This assumes that there are
				 * enough logical CPUs per LLC to avoid SMT
				 * factors and that there is a correlation
				 * between LLCs and memory channels.
				 */
				nr_llcs = sd->span_weight / child->span_weight;
				if (nr_llcs == 1)
					imb = sd->span_weight >> 3;
				else
					imb = nr_llcs;
				imb = max(1U, imb);
				sd->imb_numa_nr = imb;

				/* Set span based on the first NUMA domain. */
				top_p = sd->parent;
				while (top_p && !(top_p->flags & SD_NUMA)) {
					top_p = top_p->parent;
				}
				/* 以首个 NUMA 域为基准，向上按范围比例放宽。 */
				imb_span = top_p ? top_p->span_weight : sd->span_weight;
			} else {
				int factor = max(1U, (sd->span_weight / imb_span));

				sd->imb_numa_nr = imb * factor;
			}
		}
	}

	/* 从底层向上汇总容量，接管存储以免收尾时误释放在用对象。 */
	/* Calculate CPU capacity for physical packages and nodes */
	for (i = nr_cpumask_bits-1; i >= 0; i--) {
		if (!cpumask_test_cpu(i, cpu_map))
			continue;

		for (sd = *per_cpu_ptr(d.sd, i); sd; sd = sd->parent) {
			claim_allocations(i, sd);
			init_sched_groups_capacity(i, sd);
		}
	}

	/* Attach the domains */
	rcu_read_lock();
	for_each_cpu(i, cpu_map) {
		rq = cpu_rq(i);
		sd = *per_cpu_ptr(d.sd, i);

		cpu_attach_domain(sd, d.rd, i);

		if (lowest_flag_domain(i, SD_CLUSTER))
			has_cluster = true;
	}
	rcu_read_unlock();

	/* 静态分支按使用特性的分区计数，拆除分区时对应递减。 */
	if (has_asym)
		static_branch_inc_cpuslocked(&sched_asym_cpucapacity);

	if (has_cluster)
		static_branch_inc_cpuslocked(&sched_cluster_active);

	if (rq && sched_debug_verbose)
		pr_info("root domain span: %*pbl\n", cpumask_pr_args(cpu_map));

	ret = 0;
error:
	__free_domain_allocs(&d, alloc_state, cpu_map);

	return ret;
}

/* doms_cur 的每个掩码代表一个不相交的均衡分区，而不是单层 sched_domain。 */
/* Current sched domains: */
static cpumask_var_t			*doms_cur;

/* Number of sched domains in 'doms_cur': */
static int				ndoms_cur;

/* Attributes of custom domains in 'doms_cur' */
static struct sched_domain_attr		*dattr_cur;

/*
 * Special case: If a kmalloc() of a doms_cur partition (array of
 * cpumask) fails, then fallback to a single sched domain,
 * as determined by the single cpumask fallback_doms.
 */
static cpumask_var_t			fallback_doms;

/*
 * arch_update_cpu_topology lets virtualized architectures update the
 * CPU core maps. It is supposed to return 1 if the topology changed
 * or 0 if it stayed the same.
 */
int __weak arch_update_cpu_topology(void)
{
	return 0;
}

/* 分配分区掩码数组；中途失败只回收此前已成功分配的掩码。 */
cpumask_var_t *alloc_sched_domains(unsigned int ndoms)
{
	int i;
	cpumask_var_t *doms;

	doms = kmalloc_array(ndoms, sizeof(*doms), GFP_KERNEL);
	if (!doms)
		return NULL;
	for (i = 0; i < ndoms; i++) {
		if (!alloc_cpumask_var(&doms[i], GFP_KERNEL)) {
			free_sched_domains(doms, i);
			return NULL;
		}
	}
	return doms;
}

/* 释放分区描述数组，不负责拆除这些掩码对应的实际调度域。 */
void free_sched_domains(cpumask_var_t doms[], unsigned int ndoms)
{
	unsigned int i;
	for (i = 0; i < ndoms; i++)
		free_cpumask_var(doms[i]);
	kfree(doms);
}

/*
 * Set up scheduler domains and groups.  For now this just excludes isolated
 * CPUs, but could be used to exclude other special cases in the future.
 *
 * 初始仅建立一个均衡分区，并通过 HK_TYPE_DOMAIN 排除不参与域均衡的 CPU。
 * 掩码数组分配失败时使用预留的单分区存储，仍尝试完成拓扑初始化。
 */
int __init sched_init_domains(const struct cpumask *cpu_map)
{
	int err;

	zalloc_cpumask_var(&sched_domains_tmpmask, GFP_KERNEL);
	zalloc_cpumask_var(&sched_domains_tmpmask2, GFP_KERNEL);
	zalloc_cpumask_var(&fallback_doms, GFP_KERNEL);

	arch_update_cpu_topology();
	asym_cpu_capacity_scan();
	ndoms_cur = 1;
	doms_cur = alloc_sched_domains(ndoms_cur);
	if (!doms_cur)
		doms_cur = &fallback_doms;
	cpumask_and(doms_cur[0], cpu_map, housekeeping_cpumask(HK_TYPE_DOMAIN));
	err = build_sched_domains(doms_cur[0], NULL);

	return err;
}

/*
 * Detach sched domains from a group of CPUs specified in cpu_map
 * These CPUs will now be attached to the NULL domain
 *
 * NULL 表示没有公平类域均衡层级，并非 CPU 离线；仍挂接默认根域。
 * cpu_attach_domain() 同时清空拓扑快捷指针并延迟销毁旧链。
 */
static void detach_destroy_domains(const struct cpumask *cpu_map)
{
	unsigned int cpu = cpumask_any(cpu_map);
	int i;

	if (rcu_access_pointer(per_cpu(sd_asym_cpucapacity, cpu)))
		static_branch_dec_cpuslocked(&sched_asym_cpucapacity);

	if (static_branch_unlikely(&sched_cluster_active))
		static_branch_dec_cpuslocked(&sched_cluster_active);

	rcu_read_lock();
	for_each_cpu(i, cpu_map)
		cpu_attach_domain(NULL, &def_root_domain, i);
	rcu_read_unlock();
}

/* handle null as "default" */
/* 缺省属性按 SD_ATTR_INIT 比较，允许显式默认值复用旧域。 */
static int dattrs_equal(struct sched_domain_attr *cur, int idx_cur,
			struct sched_domain_attr *new, int idx_new)
{
	struct sched_domain_attr tmp;

	/* Fast path: */
	if (!new && !cur)
		return 1;

	tmp = SD_ATTR_INIT;

	return !memcmp(cur ? (cur + idx_cur) : &tmp,
			new ? (new + idx_new) : &tmp,
			sizeof(struct sched_domain_attr));
}

/*
 * Partition sched domains as specified by the 'ndoms_new'
 * cpumasks in the array doms_new[] of cpumasks. This compares
 * doms_new[] to the current sched domain partitioning, doms_cur[].
 * It destroys each deleted domain and builds each new domain.
 *
 * 'doms_new' is an array of cpumask_var_t's of length 'ndoms_new'.
 * The masks don't intersect (don't overlap.) We should setup one
 * sched domain for each mask. CPUs not in any of the cpumasks will
 * not be load balanced. If the same cpumask appears both in the
 * current 'doms_cur' domains and in the new 'doms_new', we can leave
 * it as it is.
 *
 * The passed in 'doms_new' should be allocated using
 * alloc_sched_domains.  This routine takes ownership of it and will
 * free_sched_domains it when done with it. If the caller failed the
 * alloc call, then it can pass in doms_new == NULL && ndoms_new == 1,
 * and partition_sched_domains() will fallback to the single partition
 * 'fallback_doms', it also forces the domains to be rebuilt.
 *
 * If doms_new == NULL it will be replaced with cpu_online_mask.
 * ndoms_new == 0 is a special case for destroying existing domains,
 * and it will not create the default domain.
 *
 * Call with hotplug lock and sched_domains_mutex held
 *
 * 增量更新分三轮：拆除失配旧分区、创建新分区、刷新 EAS 性能域。
 * CPU 掩码与属性均相等才复用调度域；架构拓扑变化则强制重建。
 * doms_new/dattr_new 的所有权在此转移，调用者不能再自行释放或修改它们。
 */
static void partition_sched_domains_locked(int ndoms_new, cpumask_var_t doms_new[],
				    struct sched_domain_attr *dattr_new)
{
	bool __maybe_unused has_eas = false;
	int i, j, n;
	int new_topology;

	lockdep_assert_held(&sched_domains_mutex);

	/* Let the architecture update CPU core mappings: */
	new_topology = arch_update_cpu_topology();
	/* Trigger rebuilding CPU capacity asymmetry data */
	if (new_topology)
		asym_cpu_capacity_scan();

	/* 未提供掩码时尝试构造 active 且参与域均衡的默认集合。 */
	if (!doms_new) {
		WARN_ON_ONCE(dattr_new);
		n = 0;
		doms_new = alloc_sched_domains(1);
		if (doms_new) {
			n = 1;
			cpumask_and(doms_new[0], cpu_active_mask,
				    housekeeping_cpumask(HK_TYPE_DOMAIN));
		}
	} else {
		n = ndoms_new;
	}

	/* Destroy deleted domains: */
	for (i = 0; i < ndoms_cur; i++) {
		for (j = 0; j < n && !new_topology; j++) {
			if (cpumask_equal(doms_cur[i], doms_new[j]) &&
			    dattrs_equal(dattr_cur, i, dattr_new, j))
				goto match1;
		}
		/* No match - a current sched domain not in new doms_new[] */
		detach_destroy_domains(doms_cur[i]);
match1:
		;
	}

	n = ndoms_cur;
	if (!doms_new) {
		/* 分配仍失败时使用预留存储，不再匹配已拆除的旧域。 */
		n = 0;
		doms_new = &fallback_doms;
		cpumask_and(doms_new[0], cpu_active_mask,
			    housekeeping_cpumask(HK_TYPE_DOMAIN));
	}

	/* Build new domains: */
	for (i = 0; i < ndoms_new; i++) {
		for (j = 0; j < n && !new_topology; j++) {
			if (cpumask_equal(doms_new[i], doms_cur[j]) &&
			    dattrs_equal(dattr_new, i, dattr_cur, j))
				goto match2;
		}
		/* No match - add a new doms_new */
		build_sched_domains(doms_new[i], dattr_new ? dattr_new + i : NULL);
match2:
		;
	}

#if defined(CONFIG_ENERGY_MODEL) && defined(CONFIG_CPU_FREQ_GOV_SCHEDUTIL)
	/* 即使 CPU 分区不变，显式能耗更新仍要求重建性能域。 */
	/* Build perf domains: */
	for (i = 0; i < ndoms_new; i++) {
		for (j = 0; j < n && !sched_energy_update; j++) {
			if (cpumask_equal(doms_new[i], doms_cur[j]) &&
			    cpu_rq(cpumask_first(doms_cur[j]))->rd->pd) {
				has_eas = true;
				goto match3;
			}
		}
		/* No match - add perf domains for a new rd */
		has_eas |= build_perf_domains(doms_new[i]);
match3:
		;
	}
	sched_energy_set(has_eas);
#endif

	/* Remember the new sched domains: */
	if (doms_cur != &fallback_doms)
		free_sched_domains(doms_cur, ndoms_cur);

	kfree(dattr_cur);
	doms_cur = doms_new;
	dattr_cur = dattr_new;
	ndoms_cur = ndoms_new;

	/* 域归属变化后，更新调试视图并重建 DL 根域带宽记账。 */
	update_sched_domain_debugfs();
	dl_rebuild_rd_accounting();
}

/*
 * Call with hotplug lock held
 *
 * 对外的分区重建入口；另取域互斥锁保护共享临时掩码和分区描述，
 * 在 CPU 热插拔锁的约束下串行化域链替换过程。
 */
void partition_sched_domains(int ndoms_new, cpumask_var_t doms_new[],
			     struct sched_domain_attr *dattr_new)
{
	sched_domains_mutex_lock();
	partition_sched_domains_locked(ndoms_new, doms_new, dattr_new);
	sched_domains_mutex_unlock();
}
