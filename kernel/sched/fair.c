// SPDX-License-Identifier: GPL-2.0
/*
 * Completely Fair Scheduling (CFS) Class (SCHED_NORMAL/SCHED_BATCH)
 *
 *  Copyright (C) 2007 Red Hat, Inc., Ingo Molnar <mingo@redhat.com>
 *
 *  Interactivity improvements by Mike Galbraith
 *  (C) 2007 Mike Galbraith <efault@gmx.de>
 *
 *  Various enhancements by Dmitry Adamushko.
 *  (C) 2007 Dmitry Adamushko <dmitry.adamushko@gmail.com>
 *
 *  Group scheduling enhancements by Srivatsa Vaddagiri
 *  Copyright IBM Corporation, 2007
 *  Author: Srivatsa Vaddagiri <vatsa@linux.vnet.ibm.com>
 *
 *  Scaled math optimizations by Thomas Gleixner
 *  Copyright (C) 2007, Thomas Gleixner <tglx@linutronix.de>
 *
 *  Adaptive scheduling granularity, math enhancements by Peter Zijlstra
 *  Copyright (C) 2007 Red Hat, Inc., Peter Zijlstra
 */
/*
 * 公平调度类阅读导航：
 *
 * 本文件保留 CFS 命名，当前实体选择使用 EEVDF：先判断是否尚欠服务，
 * 再比较虚拟截止时间；这是公平类内部的排序规则，
 * 不提供实时截止期保证。
 *
 * 主要路径：
 *   入队：enqueue_task_fair() -> enqueue_entity() -> place_entity()。
 *   选取：pick_next_task_fair() -> pick_task_fair() -> pick_next_entity()
 *         -> pick_eevdf()；组调度从根队列逐层向下选到任务。
 *   记账：task_tick_fair() -> entity_tick() -> update_curr()。
 *   出队：dequeue_task_fair() -> dequeue_entities() -> dequeue_entity()。
 *   选核：select_task_rq_fair() 综合唤醒亲和性、空闲容量及能耗模型。
 *   均衡：sched_balance_rq() 选择源组/源队列，摘取任务后接入目标队列。
 *
 * 数据层级：rq 是每 CPU 总运行队列；cfs_rq 是某一层公平竞争队列；
 * sched_entity 可代表任务，也可代表一个任务组在父队列中的竞争者。
 * 任务数使用 h_nr_* 沿层级汇总，nr_queued 则统计本层排队实体数。
 *
 * 关键状态：on_rq 表示实体仍计入公平队列，不保证它位于红黑树上。
 * 当前实体单独保存在 cfs_rq->curr；sched_delayed 实体虽暂留队列，
 * 也不再算作可运行任务。使用这些状态时不能只凭是否在树中判断。
 *
 * 阅读顺序可从虚拟时间/EEVDF 开始，再看实体入出队和调度类回调；
 * PELT、NUMA、带宽控制、能耗选核与多 CPU 均衡分别提供统计和策略支持。
 */
#include <linux/energy_model.h>
#include <linux/mmap_lock.h>
#include <linux/hugetlb_inline.h>
#include <linux/jiffies.h>
#include <linux/mm_api.h>
#include <linux/highmem.h>
#include <linux/spinlock_api.h>
#include <linux/cpumask_api.h>
#include <linux/lockdep_api.h>
#include <linux/softirq.h>
#include <linux/refcount_api.h>
#include <linux/topology.h>
#include <linux/sched/clock.h>
#include <linux/sched/cond_resched.h>
#include <linux/sched/cputime.h>
#include <linux/sched/isolation.h>
#include <linux/sched/nohz.h>
#include <linux/sched/prio.h>

#include <linux/cpuidle.h>
#include <linux/interrupt.h>
#include <linux/memory-tiers.h>
#include <linux/mempolicy.h>
#include <linux/mutex_api.h>
#include <linux/profile.h>
#include <linux/psi.h>
#include <linux/ratelimit.h>
#include <linux/task_work.h>
#include <linux/rbtree_augmented.h>

#include <asm/switch_to.h>

#include <uapi/linux/sched/types.h>

#include "sched.h"
#include "stats.h"
#include "autogroup.h"

/*
 * The initial- and re-scaling of tunables is configurable
 *
 * Options are:
 *
 *   SCHED_TUNABLESCALING_NONE - unscaled, always *1
 *   SCHED_TUNABLESCALING_LOG - scaled logarithmically, *1+ilog(ncpus)
 *   SCHED_TUNABLESCALING_LINEAR - scaled linear, *ncpus
 *
 * (default SCHED_TUNABLESCALING_LOG = *(1+ilog(ncpus))
 */
unsigned int sysctl_sched_tunable_scaling = SCHED_TUNABLESCALING_LOG;

/*
 * Minimal preemption granularity for CPU-bound tasks:
 *
 * (default: 0.70 msec * (1 + ilog(ncpus)), units: nanoseconds)
 */
unsigned int sysctl_sched_base_slice			= 700000ULL;
static unsigned int normalized_sysctl_sched_base_slice	= 700000ULL;

__read_mostly unsigned int sysctl_sched_migration_cost	= 500000UL;

/*
 * setup_sched_thermal_decay_shift - 兼容旧启动参数；仅提示参数已废弃，
 * 不再调整热压力衰减。
 */
static int __init setup_sched_thermal_decay_shift(char *str)
{
	pr_warn("Ignoring the deprecated sched_thermal_decay_shift= option\n");
	return 1;
}
__setup("sched_thermal_decay_shift=", setup_sched_thermal_decay_shift);

/*
 * For asym packing, by default the lower numbered CPU has higher priority.
 */
/*
 * arch_asym_cpu_priority - 给非对称打包提供默认 CPU 优先级，
 * 编号越小优先级越高。
 */
int __weak arch_asym_cpu_priority(int cpu)
{
	return -cpu;
}

/*
 * The margin used when comparing utilization with CPU capacity.
 *
 * (default: ~20%)
 */
#define fits_capacity(cap, max)	((cap) * 1280 < (max) * 1024)

/*
 * The margin used when comparing CPU capacities.
 * is 'cap1' noticeably greater than 'cap2'
 *
 * (default: ~5%)
 */
#define capacity_greater(cap1, cap2) ((cap1) * 1024 > (cap2) * 1078)

#ifdef CONFIG_CFS_BANDWIDTH
/*
 * Amount of runtime to allocate from global (tg) to local (per-cfs_rq) pool
 * each time a cfs_rq requests quota.
 *
 * Note: in the case that the slice exceeds the runtime remaining (either due
 * to consumption or the quota being specified to be smaller than the slice)
 * we will always only issue the remaining available time.
 *
 * (default: 5 msec, units: microseconds)
 */
static unsigned int sysctl_sched_cfs_bandwidth_slice		= 5000UL;
#endif

#ifdef CONFIG_NUMA_BALANCING
/* Restrict the NUMA promotion throughput (MB/s) for each target node. */
static unsigned int sysctl_numa_balancing_promote_rate_limit = 65536;
#endif

#ifdef CONFIG_SYSCTL
static const struct ctl_table sched_fair_sysctls[] = {
#ifdef CONFIG_CFS_BANDWIDTH
	{
		.procname       = "sched_cfs_bandwidth_slice_us",
		.data           = &sysctl_sched_cfs_bandwidth_slice,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = proc_dointvec_minmax,
		.extra1         = SYSCTL_ONE,
	},
#endif
#ifdef CONFIG_NUMA_BALANCING
	{
		.procname	= "numa_balancing_promote_rate_limit_MBps",
		.data		= &sysctl_numa_balancing_promote_rate_limit,
		.maxlen		= sizeof(unsigned int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
	},
#endif /* CONFIG_NUMA_BALANCING */
};

/*
 * sched_fair_sysctl_init - 注册公平调度相关的 sysctl 参数。
 */
static int __init sched_fair_sysctl_init(void)
{
	register_sysctl_init("kernel", sched_fair_sysctls);
	return 0;
}
late_initcall(sched_fair_sysctl_init);
#endif /* CONFIG_SYSCTL */

/*
 * update_load_add - 增加权重并使倒数缓存失效，留待下次计算时重建。
 */
static inline void update_load_add(struct load_weight *lw, unsigned long inc)
{
	lw->weight += inc;
	lw->inv_weight = 0;
}

/*
 * update_load_sub - 减少权重并使倒数缓存失效。
 */
static inline void update_load_sub(struct load_weight *lw, unsigned long dec)
{
	lw->weight -= dec;
	lw->inv_weight = 0;
}

/*
 * update_load_set - 替换权重并清除与旧权重对应的倒数缓存。
 */
static inline void update_load_set(struct load_weight *lw, unsigned long w)
{
	lw->weight = w;
	lw->inv_weight = 0;
}

/*
 * Increase the granularity value when there are more CPUs,
 * because with more CPUs the 'effective latency' as visible
 * to users decreases. But the relationship is not linear,
 * so pick a second-best guess by going with the log2 of the
 * number of CPUs.
 *
 * This idea comes from the SD scheduler of Con Kolivas:
 */
/*
 * get_update_sysctl_factor - 根据在线 CPU 数和缩放模式计算时间片缩放因子。
 * 参与缩放的 CPU 数最多取 8，避免大系统无限放大调度粒度。
 */
static unsigned int get_update_sysctl_factor(void)
{
	unsigned int cpus = min_t(unsigned int, num_online_cpus(), 8);
	unsigned int factor;

	switch (sysctl_sched_tunable_scaling) {
	case SCHED_TUNABLESCALING_NONE:
		factor = 1;
		break;
	case SCHED_TUNABLESCALING_LINEAR:
		factor = cpus;
		break;
	case SCHED_TUNABLESCALING_LOG:
	default:
		factor = 1 + ilog2(cpus);
		break;
	}

	return factor;
}

/*
 * update_sysctl - 将归一化的基础时间片乘以当前 CPU 规模对应的因子。
 */
static void update_sysctl(void)
{
	unsigned int factor = get_update_sysctl_factor();

#define SET_SYSCTL(name) \
	(sysctl_##name = (factor) * normalized_sysctl_##name)
	SET_SYSCTL(sched_base_slice);
#undef SET_SYSCTL
}

/*
 * sched_init_granularity - 在初始化时应用公平调度的时间粒度设置。
 */
void __init sched_init_granularity(void)
{
	update_sysctl();
}

#define WMULT_CONST	(~0U)
#define WMULT_SHIFT	32

/*
 * __update_inv_weight - 按需缓存权重倒数，用定点乘法代替后续路径中的除法。
 */
static void __update_inv_weight(struct load_weight *lw)
{
	unsigned long w;

	if (likely(lw->inv_weight))
		return;

	w = scale_load_down(lw->weight);

	if (BITS_PER_LONG > 32 && unlikely(w >= WMULT_CONST))
		lw->inv_weight = 1;
	else if (unlikely(!w))
		lw->inv_weight = WMULT_CONST;
	else
		lw->inv_weight = WMULT_CONST / w;
}

/*
 * delta_exec * weight / lw.weight
 *   OR
 * (delta_exec * (weight * lw->inv_weight)) >> WMULT_SHIFT
 *
 * Either weight := NICE_0_LOAD and lw \e sched_prio_to_wmult[], in which case
 * we're guaranteed shift stays positive because inv_weight is guaranteed to
 * fit 32 bits, and NICE_0_LOAD gives another 10 bits; therefore shift >= 22.
 *
 * Or, weight =< lw.weight (because lw.weight is the runqueue weight), thus
 * weight/lw.weight <= 1, and therefore our shift will also be positive.
 */
/*
 * __calc_delta - 计算 delta_exec * weight / lw->weight。
 * 通过调整乘数和移位量，将中间乘数限制在 32 位范围内。
 */
static u64 __calc_delta(u64 delta_exec, unsigned long weight, struct load_weight *lw)
{
	u64 fact = scale_load_down(weight);
	u32 fact_hi = (u32)(fact >> 32);
	int shift = WMULT_SHIFT;
	int fs;

	__update_inv_weight(lw);

	if (unlikely(fact_hi)) {
		fs = fls(fact_hi);
		shift -= fs;
		fact >>= fs;
	}

	fact = mul_u32_u32(fact, lw->inv_weight);

	fact_hi = (u32)(fact >> 32);
	if (fact_hi) {
		fs = fls(fact_hi);
		shift -= fs;
		fact >>= fs;
	}

	return mul_u64_u32_shr(delta_exec, fact, shift);
}

/*
 * delta /= w
 */
/*
 * calc_delta_fair - 将实际执行时间换算为实体的虚拟时间。
 * 权重越大，同样实际运行时间带来的 vruntime 增量越小。
 */
static inline u64 calc_delta_fair(u64 delta, struct sched_entity *se)
{
	if (unlikely(se->load.weight != NICE_0_LOAD))
		delta = __calc_delta(delta, NICE_0_LOAD, &se->load);

	return delta;
}

const struct sched_class fair_sched_class;

/**************************************************************
 * CFS operations on generic schedulable entities:
 */

#ifdef CONFIG_FAIR_GROUP_SCHED

/* Walk up scheduling entities hierarchy */
#define for_each_sched_entity(se) \
		for (; se; se = se->parent)

/*
 * list_add_leaf_cfs_rq - 把组运行队列加入叶子遍历链表，
 * 保证子队列排在父队列之前。
 * 返回 true 表示当前分支已连接到树中，临时分支指针已归位。
 */
static inline bool list_add_leaf_cfs_rq(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	int cpu = cpu_of(rq);

	if (cfs_rq->on_list)
		return rq->tmp_alone_branch == &rq->leaf_cfs_rq_list;

	cfs_rq->on_list = 1;

	/*
	 * Ensure we either appear before our parent (if already
	 * enqueued) or force our parent to appear after us when it is
	 * enqueued. The fact that we always enqueue bottom-up
	 * reduces this to two cases and a special case for the root
	 * cfs_rq. Furthermore, it also means that we will always reset
	 * tmp_alone_branch either when the branch is connected
	 * to a tree or when we reach the top of the tree
	 */
	if (cfs_rq->tg->parent &&
	    cfs_rq->tg->parent->cfs_rq[cpu]->on_list) {
		/*
		 * If parent is already on the list, we add the child
		 * just before. Thanks to circular linked property of
		 * the list, this means to put the child at the tail
		 * of the list that starts by parent.
		 */
		list_add_tail_rcu(&cfs_rq->leaf_cfs_rq_list,
			&(cfs_rq->tg->parent->cfs_rq[cpu]->leaf_cfs_rq_list));
		/*
		 * The branch is now connected to its tree so we can
		 * reset tmp_alone_branch to the beginning of the
		 * list.
		 */
		rq->tmp_alone_branch = &rq->leaf_cfs_rq_list;
		return true;
	}

	if (!cfs_rq->tg->parent) {
		/*
		 * cfs rq without parent should be put
		 * at the tail of the list.
		 */
		list_add_tail_rcu(&cfs_rq->leaf_cfs_rq_list,
			&rq->leaf_cfs_rq_list);
		/*
		 * We have reach the top of a tree so we can reset
		 * tmp_alone_branch to the beginning of the list.
		 */
		rq->tmp_alone_branch = &rq->leaf_cfs_rq_list;
		return true;
	}

	/*
	 * The parent has not already been added so we want to
	 * make sure that it will be put after us.
	 * tmp_alone_branch points to the begin of the branch
	 * where we will add parent.
	 */
	list_add_rcu(&cfs_rq->leaf_cfs_rq_list, rq->tmp_alone_branch);
	/*
	 * update tmp_alone_branch to points to the new begin
	 * of the branch
	 */
	rq->tmp_alone_branch = &cfs_rq->leaf_cfs_rq_list;
	return false;
}

/*
 * list_del_leaf_cfs_rq - 移除叶子队列，并修正可能仍指向它的临时分支位置。
 */
static inline void list_del_leaf_cfs_rq(struct cfs_rq *cfs_rq)
{
	if (cfs_rq->on_list) {
		struct rq *rq = rq_of(cfs_rq);

		/*
		 * With cfs_rq being unthrottled/throttled during an enqueue,
		 * it can happen the tmp_alone_branch points to the leaf that
		 * we finally want to delete. In this case, tmp_alone_branch moves
		 * to the prev element but it will point to rq->leaf_cfs_rq_list
		 * at the end of the enqueue.
		 */
		if (rq->tmp_alone_branch == &cfs_rq->leaf_cfs_rq_list)
			rq->tmp_alone_branch = cfs_rq->leaf_cfs_rq_list.prev;

		list_del_rcu(&cfs_rq->leaf_cfs_rq_list);
		cfs_rq->on_list = 0;
	}
}

/*
 * assert_list_leaf_cfs_rq：
 * 检查自底向上的入队过程是否留下未连接完成的分支。
 */
static inline void assert_list_leaf_cfs_rq(struct rq *rq)
{
	WARN_ON_ONCE(rq->tmp_alone_branch != &rq->leaf_cfs_rq_list);
}

/* Iterate through all leaf cfs_rq's on a runqueue */
#define for_each_leaf_cfs_rq_safe(rq, cfs_rq, pos)			\
	list_for_each_entry_safe(cfs_rq, pos, &rq->leaf_cfs_rq_list,	\
				 leaf_cfs_rq_list)

/* Do the two (enqueued) entities belong to the same group ? */
/*
 * is_same_group - 判断两个实体是否在同一 cfs_rq 上竞争，是则返回该队列。
 */
static inline struct cfs_rq *
is_same_group(struct sched_entity *se, struct sched_entity *pse)
{
	if (se->cfs_rq == pse->cfs_rq)
		return se->cfs_rq;

	return NULL;
}

/*
 * parent_entity - 取得上一层任务组在父队列中的调度实体。
 */
static inline struct sched_entity *parent_entity(const struct sched_entity *se)
{
	return se->parent;
}

/*
 * find_matching_se - 沿组层级上行，找到处于同一 cfs_rq 的两个祖先实体。
 * 只有在同一竞争层级，虚拟截止时间和抢占关系才可直接比较。
 */
static void
find_matching_se(struct sched_entity **se, struct sched_entity **pse)
{
	int se_depth, pse_depth;

	/*
	 * preemption test can be made between sibling entities who are in the
	 * same cfs_rq i.e who have a common parent. Walk up the hierarchy of
	 * both tasks until we find their ancestors who are siblings of common
	 * parent.
	 */

	/* First walk up until both entities are at same depth */
	se_depth = (*se)->depth;
	pse_depth = (*pse)->depth;

	while (se_depth > pse_depth) {
		se_depth--;
		*se = parent_entity(*se);
	}

	while (pse_depth > se_depth) {
		pse_depth--;
		*pse = parent_entity(*pse);
	}

	while (!is_same_group(*se, *pse)) {
		*se = parent_entity(*se);
		*pse = parent_entity(*pse);
	}
}

/*
 * tg_is_idle - 判断任务组是否采用空闲调度属性。
 */
static int tg_is_idle(struct task_group *tg)
{
	return tg->idle > 0;
}

/*
 * cfs_rq_is_idle - 判断该组运行队列是否标记为空闲调度组。
 */
static int cfs_rq_is_idle(struct cfs_rq *cfs_rq)
{
	return cfs_rq->idle > 0;
}

/*
 * se_is_idle - 统一判断任务实体的 SCHED_IDLE 策略和组实体的 idle 属性。
 */
static int se_is_idle(struct sched_entity *se)
{
	if (entity_is_task(se))
		return task_has_idle_policy(task_of(se));
	return cfs_rq_is_idle(group_cfs_rq(se));
}

#else /* !CONFIG_FAIR_GROUP_SCHED: */

#define for_each_sched_entity(se) \
		for (; se; se = NULL)

/*
 * list_add_leaf_cfs_rq - 未启用组调度时无需维护叶子链表，
 * 直接视为连接完成。
 */
static inline bool list_add_leaf_cfs_rq(struct cfs_rq *cfs_rq)
{
	return true;
}

/*
 * list_del_leaf_cfs_rq - 未启用组调度时，无叶子链表需要移除。
 */
static inline void list_del_leaf_cfs_rq(struct cfs_rq *cfs_rq)
{
}

/*
 * assert_list_leaf_cfs_rq - 未启用组调度时，无层级链表不变量需要检查。
 */
static inline void assert_list_leaf_cfs_rq(struct rq *rq)
{
}

#define for_each_leaf_cfs_rq_safe(rq, cfs_rq, pos)	\
		for (cfs_rq = &rq->cfs, pos = NULL; cfs_rq; cfs_rq = pos)

/*
 * parent_entity - 未启用组调度时只有任务层，没有父实体。
 */
static inline struct sched_entity *parent_entity(struct sched_entity *se)
{
	return NULL;
}

/*
 * find_matching_se - 未启用组调度时，同一 CPU 上的实体已在同一竞争层级。
 */
static inline void
find_matching_se(struct sched_entity **se, struct sched_entity **pse)
{
}

/*
 * tg_is_idle - 未启用组调度时不存在组级 idle 属性。
 */
static inline int tg_is_idle(struct task_group *tg)
{
	return 0;
}

/*
 * cfs_rq_is_idle - 未启用组调度时根队列不具有组级 idle 属性。
 */
static int cfs_rq_is_idle(struct cfs_rq *cfs_rq)
{
	return 0;
}

/*
 * se_is_idle - 无组层级时，仅检查任务自身是否采用 SCHED_IDLE 策略。
 */
static int se_is_idle(struct sched_entity *se)
{
	return task_has_idle_policy(task_of(se));
}

#endif /* !CONFIG_FAIR_GROUP_SCHED */

static __always_inline
void account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec);

/**************************************************************
 * Scheduling class tree data structure manipulation methods:
 */

extern void __BUILD_BUG_vruntime_cmp(void);

/* Use __builtin_strcmp() because of __HAVE_ARCH_STRCMP: */

/*
 * 虚拟时间使用无符号计数保存，通过相减后转为 s64 比较相对先后。
 * 在差值处于可表示范围的前提下，这种方式可跨越 u64 回绕点。
 * 下方红黑树按 deadline 排序，min_vruntime 仅作为子树资格查询的增强字段。
 */
#define vruntime_cmp(A, CMP_STR, B) ({				\
	int __res = 0;						\
								\
	if (!__builtin_strcmp(CMP_STR, "<")) {			\
		__res = ((s64)((A)-(B)) < 0);			\
	} else if (!__builtin_strcmp(CMP_STR, "<=")) {		\
		__res = ((s64)((A)-(B)) <= 0);			\
	} else if (!__builtin_strcmp(CMP_STR, ">")) {		\
		__res = ((s64)((A)-(B)) > 0);			\
	} else if (!__builtin_strcmp(CMP_STR, ">=")) {		\
		__res = ((s64)((A)-(B)) >= 0);			\
	} else {						\
		/* Unknown operator throws linker error: */	\
		__BUILD_BUG_vruntime_cmp();			\
	}							\
								\
	__res;							\
})

extern void __BUILD_BUG_vruntime_op(void);

#define vruntime_op(A, OP_STR, B) ({				\
	s64 __res = 0;						\
								\
	if (!__builtin_strcmp(OP_STR, "-")) {			\
		__res = (s64)((A)-(B));				\
	} else {						\
		/* Unknown operator throws linker error: */	\
		__BUILD_BUG_vruntime_op();			\
	}							\
								\
	__res;						\
})


/*
 * max_vruntime - 按带符号的相对差值选较大的虚拟时间，兼容计数器回绕。
 */
static inline __maybe_unused u64 max_vruntime(u64 max_vruntime, u64 vruntime)
{
	if (vruntime_cmp(vruntime, ">", max_vruntime))
		max_vruntime = vruntime;

	return max_vruntime;
}

/*
 * min_vruntime - 按带符号的相对差值选较小的虚拟时间。
 */
static inline __maybe_unused u64 min_vruntime(u64 min_vruntime, u64 vruntime)
{
	if (vruntime_cmp(vruntime, "<", min_vruntime))
		min_vruntime = vruntime;

	return min_vruntime;
}

/*
 * entity_before - 按虚拟截止时间排序实体；这里的 deadline 不是墙上时间。
 */
static inline bool entity_before(const struct sched_entity *a,
				 const struct sched_entity *b)
{
	/*
	 * Tiebreak on vruntime seems unnecessary since it can
	 * hardly happen.
	 */
	return vruntime_cmp(a->deadline, "<", b->deadline);
}

/*
 * Per avg_vruntime() below, cfs_rq::zero_vruntime is only slightly stale
 * and this value should be no more than two lag bounds. Which puts it in the
 * general order of:
 *
 *	(slice + TICK_NSEC) << NICE_0_LOAD_SHIFT
 *
 * which is around 44 bits in size (on 64bit); that is 20 for
 * NICE_0_LOAD_SHIFT, another 20 for NSEC_PER_MSEC and then a handful for
 * however many msec the actual slice+tick ends up begin.
 *
 * (disregarding the actual divide-by-weight part makes for the worst case
 * weight of 2, which nicely cancels vs the fuzz in zero_vruntime not actually
 * being the zero-lag point).
 */
/*
 * entity_key - 取实体相对 zero_vruntime 的偏移，降低加权求和溢出的风险。
 */
static inline s64 entity_key(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	return vruntime_op(se->vruntime, "-", cfs_rq->zero_vruntime);
}

#define __node_2_se(node) \
	rb_entry((node), struct sched_entity, run_node)

/*
 * Compute virtual time from the per-task service numbers:
 *
 * Fair schedulers conserve lag:
 *
 *   \Sum lag_i = 0
 *
 * Where lag_i is given by:
 *
 *   lag_i = S - s_i = w_i * (V - v_i)
 *
 * Where S is the ideal service time and V is it's virtual time counterpart.
 * Therefore:
 *
 *   \Sum lag_i = 0
 *   \Sum w_i * (V - v_i) = 0
 *   \Sum w_i * V - w_i * v_i = 0
 *
 * From which we can solve an expression for V in v_i (which we have in
 * se->vruntime):
 *
 *       \Sum v_i * w_i   \Sum v_i * w_i
 *   V = -------------- = --------------
 *          \Sum w_i            W
 *
 * Specifically, this is the weighted average of all entity virtual runtimes.
 *
 * [[ NOTE: this is only equal to the ideal scheduler under the condition
 *          that join/leave operations happen at lag_i = 0, otherwise the
 *          virtual time has non-contiguous motion equivalent to:
 *
 *	      V +-= lag_i / W
 *
 *	    Also see the comment in place_entity() that deals with this. ]]
 *
 * However, since v_i is u64, and the multiplication could easily overflow
 * transform it into a relative form that uses smaller quantities:
 *
 * Substitute: v_i == (v_i - v0) + v0
 *
 *     \Sum ((v_i - v0) + v0) * w_i   \Sum (v_i - v0) * w_i
 * V = ---------------------------- = --------------------- + v0
 *                  W                            W
 *
 * Which we track using:
 *
 *                    v0 := cfs_rq->zero_vruntime
 * \Sum (v_i - v0) * w_i := cfs_rq->sum_w_vruntime
 *              \Sum w_i := cfs_rq->sum_weight
 *
 * Since zero_vruntime closely tracks the per-task service, these
 * deltas: (v_i - v), will be in the order of the maximal (virtual) lag
 * induced in the system due to quantisation.
 *
 * Also, we use scale_load_down() to reduce the size.
 *
 * As measured, the max (key * weight) value was ~44 bits for a kernel build.
 */
/*
 * sum_w_vruntime_add - 将树中实体的加权虚拟时间和权重加入队列累计值。
 */
static void
sum_w_vruntime_add(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	unsigned long weight = scale_load_down(se->load.weight);
	s64 key = entity_key(cfs_rq, se);

	cfs_rq->sum_w_vruntime += key * weight;
	cfs_rq->sum_weight += weight;
}

/*
 * sum_w_vruntime_sub - 从队列累计值中减去离开红黑树的实体贡献。
 */
static void
sum_w_vruntime_sub(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	unsigned long weight = scale_load_down(se->load.weight);
	s64 key = entity_key(cfs_rq, se);

	cfs_rq->sum_w_vruntime -= key * weight;
	cfs_rq->sum_weight -= weight;
}

/*
 * update_zero_vruntime - 平移虚拟时间参考原点，并反向补偿加权和。
 * 这种坐标变换不改变实体之间的相对服务关系。
 */
static inline
void update_zero_vruntime(struct cfs_rq *cfs_rq, s64 delta)
{
	/*
	 * v' = v + d ==> sum_w_vruntime' = sum_w_vruntime - d*sum_weight
	 */
	cfs_rq->sum_w_vruntime -= cfs_rq->sum_weight * delta;
	cfs_rq->zero_vruntime += delta;
}

/*
 * Specifically: avg_vruntime() + 0 must result in entity_eligible() := true
 * For this to be so, the result of this function must have a left bias.
 *
 * Called in:
 *  - place_entity()      -- before enqueue
 *  - update_entity_lag() -- before dequeue
 *  - update_deadline()   -- slice expiration
 *
 * This means it is one entry 'behind' but that puts it close enough to where
 * the bound on entity_key() is at most two lag bounds.
 */
/*
 * avg_vruntime - 计算含当前实体在内的加权平均虚拟时间，并更新参考原点。
 * 负数除法向下取整，避免零 lag 实体因舍入被误判为不具备资格。
 */
u64 avg_vruntime(struct cfs_rq *cfs_rq)
{
	struct sched_entity *curr = cfs_rq->curr;
	long weight = cfs_rq->sum_weight;
	s64 delta = 0;

	if (curr && !curr->on_rq)
		curr = NULL;

	if (weight) {
		s64 runtime = cfs_rq->sum_w_vruntime;

		if (curr) {
			unsigned long w = scale_load_down(curr->load.weight);

			runtime += entity_key(cfs_rq, curr) * w;
			weight += w;
		}

		/* sign flips effective floor / ceiling */
		/*
		 * 平均值向下取整，确保放在平均虚拟时间处的实体仍有资格。
		 */
		if (runtime < 0)
			runtime -= (weight - 1);

		delta = div_s64(runtime, weight);
	} else if (curr) {
		/*
		 * When there is but one element, it is the average.
		 */
		delta = curr->vruntime - cfs_rq->zero_vruntime;
	}

	update_zero_vruntime(cfs_rq, delta);

	return cfs_rq->zero_vruntime;
}

static inline u64 cfs_rq_max_slice(struct cfs_rq *cfs_rq);

/*
 * lag_i = S - s_i = w_i * (V - v_i)
 *
 * However, since V is approximated by the weighted average of all entities it
 * is possible -- by addition/removal/reweight to the tree -- to move V around
 * and end up with a larger lag than we started with.
 *
 * Limit this to either double the slice length with a minimum of TICK_NSEC
 * since that is the timing granularity.
 *
 * EEVDF gives the following limit for a steady state system:
 *
 *   -r_max < lag < max(r_max, q)
 */
/*
 * entity_lag - 计算虚拟服务欠账 V - vruntime，
 * 并按队列最大时间片加一 tick 限幅。
 * 正值表示尚欠服务，负值表示已超额获得服务。
 */
static s64 entity_lag(struct cfs_rq *cfs_rq, struct sched_entity *se, u64 avruntime)
{
	u64 max_slice = cfs_rq_max_slice(cfs_rq) + TICK_NSEC;
	s64 vlag, limit;

	vlag = avruntime - se->vruntime;
	limit = calc_delta_fair(max_slice, se);

	return clamp(vlag, -limit, limit);
}

/*
 * update_entity_lag - 在实体离队前保存限幅后的虚拟 lag，
 * 供重新放置或改权重使用。
 */
static void update_entity_lag(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	WARN_ON_ONCE(!se->on_rq);

	se->vlag = entity_lag(cfs_rq, se, avg_vruntime(cfs_rq));
}

/*
 * Entity is eligible once it received less service than it ought to have,
 * eg. lag >= 0.
 *
 * lag_i = S - s_i = w_i*(V - v_i)
 *
 * lag_i >= 0 -> V >= v_i
 *
 *     \Sum (v_i - v)*w_i
 * V = ------------------ + v
 *          \Sum w_i
 *
 * lag_i >= 0 -> \Sum (v_i - v)*w_i >= (v_i - v)*(\Sum w_i)
 *
 * Note: using 'avg_vruntime() > se->vruntime' is inaccurate due
 *       to the loss in precision caused by the division.
 */
/*
 * vruntime_eligible - 判断给定虚拟时间是否已获得 EEVDF 运行资格。
 * 直接交叉相乘比较 V >= vruntime，避免除法截断影响边界判断。
 */
static int vruntime_eligible(struct cfs_rq *cfs_rq, u64 vruntime)
{
	struct sched_entity *curr = cfs_rq->curr;
	s64 avg = cfs_rq->sum_w_vruntime;
	long load = cfs_rq->sum_weight;

	/*
	 * 当前实体不在树的加权和中，这里补入其贡献后再判断资格。
	 */
	if (curr && curr->on_rq) {
		unsigned long weight = scale_load_down(curr->load.weight);

		avg += entity_key(cfs_rq, curr) * weight;
		load += weight;
	}

	return avg >= vruntime_op(vruntime, "-", cfs_rq->zero_vruntime) * load;
}

/*
 * entity_eligible - 以实体的 vruntime 检查其是否尚欠服务，即 lag 非负。
 */
int entity_eligible(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	return vruntime_eligible(cfs_rq, se->vruntime);
}

/*
 * cfs_rq_min_slice - 合并树根聚合值与当前实体，取得队列中的最短时间片。
 */
static inline u64 cfs_rq_min_slice(struct cfs_rq *cfs_rq)
{
	struct sched_entity *root = __pick_root_entity(cfs_rq);
	struct sched_entity *curr = cfs_rq->curr;
	u64 min_slice = ~0ULL;

	if (curr && curr->on_rq)
		min_slice = curr->slice;

	if (root)
		min_slice = min(min_slice, root->min_slice);

	return min_slice;
}

/*
 * cfs_rq_max_slice - 合并树根聚合值与当前实体，取得队列中的最长时间片。
 */
static inline u64 cfs_rq_max_slice(struct cfs_rq *cfs_rq)
{
	struct sched_entity *root = __pick_root_entity(cfs_rq);
	struct sched_entity *curr = cfs_rq->curr;
	u64 max_slice = 0ULL;

	if (curr && curr->on_rq)
		max_slice = curr->slice;

	if (root)
		max_slice = max(max_slice, root->max_slice);

	return max_slice;
}

/*
 * __entity_less - 将红黑树节点转换为实体，按虚拟截止时间比较顺序。
 */
static inline bool __entity_less(struct rb_node *a, const struct rb_node *b)
{
	return entity_before(__node_2_se(a), __node_2_se(b));
}

/*
 * __min_vruntime_update - 用子树的最小 vruntime 更新父节点的聚合值。
 */
static inline void __min_vruntime_update(struct sched_entity *se, struct rb_node *node)
{
	if (node) {
		struct sched_entity *rse = __node_2_se(node);

		if (vruntime_cmp(se->min_vruntime, ">", rse->min_vruntime))
			se->min_vruntime = rse->min_vruntime;
	}
}

/*
 * __min_slice_update - 用子树的最短时间片更新父节点的聚合值。
 */
static inline void __min_slice_update(struct sched_entity *se, struct rb_node *node)
{
	if (node) {
		struct sched_entity *rse = __node_2_se(node);
		if (rse->min_slice < se->min_slice)
			se->min_slice = rse->min_slice;
	}
}

/*
 * __max_slice_update - 用子树的最长时间片更新父节点的聚合值。
 */
static inline void __max_slice_update(struct sched_entity *se, struct rb_node *node)
{
	if (node) {
		struct sched_entity *rse = __node_2_se(node);
		if (rse->max_slice > se->max_slice)
			se->max_slice = rse->max_slice;
	}
}

/*
 * min_vruntime_copy：
 * 在红黑树节点替换时复制子树的虚拟时间和时间片聚合信息。
 */
static inline void min_vruntime_copy(struct sched_entity *new, struct sched_entity *old)
{
	new->min_vruntime = old->min_vruntime;
	new->min_slice = old->min_slice;
	new->max_slice = old->max_slice;
}

/*
 * se->min_vruntime = min(se->vruntime, {left,right}->min_vruntime)
 */
/*
 * min_vruntime_update - 重新汇总本节点和左右子树的增强字段。
 * 返回 true 表示聚合值未变化，可停止继续向祖先传播。
 */
static inline bool min_vruntime_update(struct sched_entity *se, bool exit)
{
	u64 old_min_vruntime = se->min_vruntime;
	u64 old_min_slice = se->min_slice;
	u64 old_max_slice = se->max_slice;
	struct rb_node *node = &se->run_node;

	se->min_vruntime = se->vruntime;
	__min_vruntime_update(se, node->rb_right);
	__min_vruntime_update(se, node->rb_left);

	se->min_slice = se->slice;
	__min_slice_update(se, node->rb_right);
	__min_slice_update(se, node->rb_left);

	se->max_slice = se->slice;
	__max_slice_update(se, node->rb_right);
	__max_slice_update(se, node->rb_left);

	return se->min_vruntime == old_min_vruntime &&
	       se->min_slice == old_min_slice &&
	       se->max_slice == old_max_slice;
}


RB_DECLARE_CALLBACKS_MULTI(static, min_vruntime_cb, struct sched_entity,
		     run_node, min_vruntime_copy, min_vruntime_update);

/*
 * Enqueue an entity into the rb-tree:
 */
/*
 * __enqueue_entity - 把实体插入按 deadline 排序的增强红黑树，并计入加权和。
 */
static void __enqueue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	sum_w_vruntime_add(cfs_rq, se);
	se->min_vruntime = se->vruntime;
	se->min_slice = se->slice;
	se->max_slice = se->slice;

	rb_add_augmented_cached(&se->run_node, &cfs_rq->tasks_timeline,
				__entity_less, &min_vruntime_cb);
}

/*
 * __dequeue_entity - 从增强红黑树摘除实体，同时移除它的加权统计贡献。
 */
static void __dequeue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	rb_erase_augmented_cached(&se->run_node, &cfs_rq->tasks_timeline,
				  &min_vruntime_cb);
	sum_w_vruntime_sub(cfs_rq, se);
}

/*
 * __pick_root_entity - 返回树根实体，常用于读取覆盖整棵树的聚合字段。
 */
struct sched_entity *__pick_root_entity(struct cfs_rq *cfs_rq)
{
	struct rb_node *root = cfs_rq->tasks_timeline.rb_root.rb_node;

	if (!root)
		return NULL;

	return __node_2_se(root);
}

/*
 * __pick_first_entity - 取虚拟截止时间最早的树节点；
 * 调用者仍需判断其运行资格。
 */
struct sched_entity *__pick_first_entity(struct cfs_rq *cfs_rq)
{
	struct rb_node *left = rb_first_cached(&cfs_rq->tasks_timeline);

	if (!left)
		return NULL;

	return __node_2_se(left);
}

/*
 * Set the vruntime up to which an entity can run before looking
 * for another entity to pick.
 * In case of run to parity, we use the shortest slice of the enqueued
 * entities to set the protected period.
 * When run to parity is disabled, we give a minimum quantum to the running
 * entity to ensure progress.
 */
/*
 * set_protect_slice - 设置当前实体的虚拟运行保护边界，避免过早切走。
 * 保护长度受自身时间片及 RUN_TO_PARITY 对应的队列最短时间片约束。
 */
static inline void set_protect_slice(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	u64 slice = normalized_sysctl_sched_base_slice;
	u64 vprot = se->deadline;

	if (sched_feat(RUN_TO_PARITY))
		slice = cfs_rq_min_slice(cfs_rq);

	slice = min(slice, se->slice);
	if (slice != se->slice)
		vprot = min_vruntime(vprot, se->vruntime + calc_delta_fair(slice, se));

	se->vprot = vprot;
}

/*
 * update_protect_slice - 按新出现的最短时间片缩短现有保护区间。
 */
static inline void update_protect_slice(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	u64 slice = cfs_rq_min_slice(cfs_rq);

	se->vprot = min_vruntime(se->vprot, se->vruntime + calc_delta_fair(slice, se));
}

/*
 * protect_slice - 判断实体是否仍处于受保护的运行区间。
 */
static inline bool protect_slice(struct sched_entity *se)
{
	return vruntime_cmp(se->vruntime, "<", se->vprot);
}

/*
 * cancel_protect_slice - 将保护边界收回到当前 vruntime，
 * 使后续选择可重新竞争。
 */
static inline void cancel_protect_slice(struct sched_entity *se)
{
	if (protect_slice(se))
		se->vprot = se->vruntime;
}

/*
 * Earliest Eligible Virtual Deadline First
 *
 * In order to provide latency guarantees for different request sizes
 * EEVDF selects the best runnable task from two criteria:
 *
 *  1) the task must be eligible (must be owed service)
 *
 *  2) from those tasks that meet 1), we select the one
 *     with the earliest virtual deadline.
 *
 * We can do this in O(log n) time due to an augmented RB-tree. The
 * tree keeps the entries sorted on deadline, but also functions as a
 * heap based on the vruntime by keeping:
 *
 *  se->min_vruntime = min(se->vruntime, se->{left,right}->min_vruntime)
 *
 * Which allows tree pruning through eligibility.
 */
/*
 * pick_eevdf - 按 EEVDF 从有资格的实体中选择虚拟截止时间最早者。
 * 同时处理 buddy 偏好和当前实体的运行保护；当前实体不在树中，
 * 需单独比较。
 */
static struct sched_entity *pick_eevdf(struct cfs_rq *cfs_rq, bool protect)
{
	struct rb_node *node = cfs_rq->tasks_timeline.rb_root.rb_node;
	struct sched_entity *se = __pick_first_entity(cfs_rq);
	struct sched_entity *curr = cfs_rq->curr;
	struct sched_entity *best = NULL;

	/*
	 * We can safely skip eligibility check if there is only one entity
	 * in this cfs_rq, saving some cycles.
	 */
	if (cfs_rq->nr_queued == 1)
		return curr && curr->on_rq ? curr : se;

	/*
	 * Picking the ->next buddy will affect latency but not fairness.
	 */
	if (sched_feat(PICK_BUDDY) &&
	    cfs_rq->next && entity_eligible(cfs_rq, cfs_rq->next)) {
		/* ->next will never be delayed */
		WARN_ON_ONCE(cfs_rq->next->sched_delayed);
		return cfs_rq->next;
	}

	if (curr && (!curr->on_rq || !entity_eligible(cfs_rq, curr)))
		curr = NULL;

	if (curr && protect && protect_slice(curr))
		return curr;

	/* Pick the leftmost entity if it's eligible */
	if (se && entity_eligible(cfs_rq, se)) {
		best = se;
		goto found;
	}

	/* Heap search for the EEVD entity */
	while (node) {
		struct rb_node *left = node->rb_left;

		/*
		 * Eligible entities in left subtree are always better
		 * choices, since they have earlier deadlines.
		 */
		/*
		 * 左子树的最小 vruntime 都不合资格时，整棵左子树即可跳过；
		 * 否则优先向左，因为那里的 deadline 更早。
		 */
		if (left && vruntime_eligible(cfs_rq,
					__node_2_se(left)->min_vruntime)) {
			node = left;
			continue;
		}

		se = __node_2_se(node);

		/*
		 * The left subtree either is empty or has no eligible
		 * entity, so check the current node since it is the one
		 * with earliest deadline that might be eligible.
		 */
		if (entity_eligible(cfs_rq, se)) {
			best = se;
			break;
		}

		node = node->rb_right;
	}
/*
 * 树搜索完成后，还需与不在树中的当前实体比较。
 */
found:
	if (!best || (curr && entity_before(curr, best)))
		best = curr;

	return best;
}

/*
 * __pick_last_entity - 取得红黑树中虚拟截止时间最晚的实体。
 */
struct sched_entity *__pick_last_entity(struct cfs_rq *cfs_rq)
{
	struct rb_node *last = rb_last(&cfs_rq->tasks_timeline.rb_root);

	if (!last)
		return NULL;

	return __node_2_se(last);
}

/**************************************************************
 * Scheduling class statistics methods:
 */
/*
 * sched_update_scaling - 把当前 sysctl 时间片折算回未按 CPU 数缩放的基准值。
 */
int sched_update_scaling(void)
{
	unsigned int factor = get_update_sysctl_factor();

#define WRT_SYSCTL(name) \
	(normalized_sysctl_##name = sysctl_##name / (factor))
	WRT_SYSCTL(sched_base_slice);
#undef WRT_SYSCTL

	return 0;
}

static void clear_buddies(struct cfs_rq *cfs_rq, struct sched_entity *se);

/*
 * XXX: strictly: vd_i += N*r_i/w_i such that: vd_i > ve_i
 * this is probably good enough.
 */
/*
 * update_deadline - 当实体耗尽本次请求时，为下一时间片计算虚拟截止时间。
 * 返回 true 表示本次请求已经用完，调用者可据此请求重调度。
 */
static bool update_deadline(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	if (vruntime_cmp(se->vruntime, "<", se->deadline))
		return false;

	/*
	 * For EEVDF the virtual time slope is determined by w_i (iow.
	 * nice) while the request time r_i is determined by
	 * sysctl_sched_base_slice.
	 */
	if (!se->custom_slice)
		se->slice = sysctl_sched_base_slice;

	/*
	 * EEVDF: vd_i = ve_i + r_i / w_i
	 */
	se->deadline = se->vruntime + calc_delta_fair(se->slice, se);
	avg_vruntime(cfs_rq);

	/*
	 * The task has consumed its request, reschedule.
	 */
	return true;
}

#include "pelt.h"

static int select_idle_sibling(struct task_struct *p, int prev_cpu, int cpu);
static unsigned long task_h_load(struct task_struct *p);
static unsigned long capacity_of(int cpu);

/* Give new sched_entity start runnable values to heavy its load in infant time */
/*
 * init_entity_runnable_average - 初始化 PELT 状态：任务从满负载权重起步，
 * 空组从零开始。
 */
void init_entity_runnable_average(struct sched_entity *se)
{
	struct sched_avg *sa = &se->avg;

	memset(sa, 0, sizeof(*sa));

	/*
	 * Tasks are initialized with full load to be seen as heavy tasks until
	 * they get a chance to stabilize to their real load level.
	 * Group entities are initialized with zero load to reflect the fact that
	 * nothing has been attached to the task group yet.
	 */
	if (entity_is_task(se))
		sa->load_avg = scale_load_down(se->load.weight);

	/* when this task is enqueued, it will contribute to its cfs_rq's load_avg */
}

/*
 * With new tasks being created, their initial util_avgs are extrapolated
 * based on the cfs_rq's current util_avg:
 *
 *   util_avg = cfs_rq->avg.util_avg / (cfs_rq->avg.load_avg + 1)
 *		* se_weight(se)
 *
 * However, in many cases, the above util_avg does not give a desired
 * value. Moreover, the sum of the util_avgs may be divergent, such
 * as when the series is a harmonic series.
 *
 * To solve this problem, we also cap the util_avg of successive tasks to
 * only 1/2 of the left utilization budget:
 *
 *   util_avg_cap = (cpu_scale - cfs_rq->avg.util_avg) / 2^n
 *
 * where n denotes the nth task and cpu_scale the CPU capacity.
 *
 * For example, for a CPU with 1024 of capacity, a simplest series from
 * the beginning would be like:
 *
 *  task  util_avg: 512, 256, 128,  64,  32,   16,    8, ...
 * cfs_rq util_avg: 512, 768, 896, 960, 992, 1008, 1016, ...
 *
 * Finally, that extrapolated util_avg is clamped to the cap (util_avg_cap)
 * if util_avg > util_avg_cap.
 */
/*
 * post_init_entity_util_avg - 用目标队列现有负载推算新任务的初始利用率。
 * 上限取剩余容量的一半，避免连续创建任务时初始估计不断膨胀。
 */
void post_init_entity_util_avg(struct task_struct *p)
{
	struct sched_entity *se = &p->se;
	struct cfs_rq *cfs_rq = cfs_rq_of(se);
	struct sched_avg *sa = &se->avg;
	long cpu_scale = arch_scale_cpu_capacity(cpu_of(rq_of(cfs_rq)));
	long cap = (long)(cpu_scale - cfs_rq->avg.util_avg) / 2;

	if (p->sched_class != &fair_sched_class) {
		/*
		 * For !fair tasks do:
		 *
		update_cfs_rq_load_avg(now, cfs_rq);
		attach_entity_load_avg(cfs_rq, se);
		switched_from_fair(rq, p);
		 *
		 * such that the next switched_to_fair() has the
		 * expected state.
		 */
		se->avg.last_update_time = cfs_rq_clock_pelt(cfs_rq);
		return;
	}

	if (cap > 0) {
		if (cfs_rq->avg.util_avg != 0) {
			sa->util_avg  = cfs_rq->avg.util_avg * se_weight(se);
			sa->util_avg /= (cfs_rq->avg.load_avg + 1);

			if (sa->util_avg > cap)
				sa->util_avg = cap;
		} else {
			sa->util_avg = cap;
		}
	}

	sa->runnable_avg = sa->util_avg;
}

/*
 * update_se - 结算自 exec_start 以来的实际执行时间。
 * 代理执行时，任务 CPU 时间记给实际执行者，
 * 组实体时间记给资源提供方。
 */
static s64 update_se(struct rq *rq, struct sched_entity *se)
{
	u64 now = rq_clock_task(rq);
	s64 delta_exec;

	delta_exec = now - se->exec_start;
	if (unlikely(delta_exec <= 0))
		return delta_exec;

	se->exec_start = now;
	if (entity_is_task(se)) {
		struct task_struct *running = rq->curr;
		/*
		 * If se is a task, we account the time against the running
		 * task, as w/ proxy-exec they may not be the same.
		 */
		running->se.exec_start = now;
		running->se.sum_exec_runtime += delta_exec;

		trace_sched_stat_runtime(running, delta_exec);
		account_group_exec_runtime(running, delta_exec);

		cgroup_account_cputime(running, delta_exec);
	} else {
		/* If not task, account the time against donor se  */
		se->sum_exec_runtime += delta_exec;
	}

	if (schedstat_enabled()) {
		struct sched_statistics *stats;

		stats = __schedstats_from_se(se);
		__schedstat_set(stats->exec_max,
				max(delta_exec, stats->exec_max));
	}

	return delta_exec;
}

static void set_next_buddy(struct sched_entity *se);

/*
 * Used by other classes to account runtime.
 */
/*
 * update_curr_common - 为其他调度类提供共用的实际运行时间记账入口。
 */
s64 update_curr_common(struct rq *rq)
{
	return update_se(rq, &rq->donor->se);
}

/*
 * Update the current task's runtime statistics.
 */
/*
 * update_curr - 结算当前公平实体的执行时间、vruntime、截止时间和带宽消耗。
 * 这里的 curr 对应调度资源提供者，代理执行时可能不同于 rq->curr。
 */
static void update_curr(struct cfs_rq *cfs_rq)
{
	/*
	 * Note: cfs_rq->curr corresponds to the task picked to
	 * run (ie: rq->donor.se) which due to proxy-exec may
	 * not necessarily be the actual task running
	 * (rq->curr.se). This is easy to confuse!
	 */
	struct sched_entity *curr = cfs_rq->curr;
	struct rq *rq = rq_of(cfs_rq);
	s64 delta_exec;
	bool resched;

	if (unlikely(!curr))
		return;

	delta_exec = update_se(rq, curr);
	if (unlikely(delta_exec <= 0))
		return;

	/*
	 * 实际执行量用于 CPU 时间和配额记账；
	 * 换算后的虚拟量用于公平竞争。
	 */
	curr->vruntime += calc_delta_fair(delta_exec, curr);
	resched = update_deadline(cfs_rq, curr);

	if (entity_is_task(curr)) {
		/*
		 * If the fair_server is active, we need to account for the
		 * fair_server time whether or not the task is running on
		 * behalf of fair_server or not:
		 *  - If the task is running on behalf of fair_server, we need
		 *    to limit its time based on the assigned runtime.
		 *  - Fair task that runs outside of fair_server should account
		 *    against fair_server such that it can account for this time
		 *    and possibly avoid running this period.
		 */
		if (dl_server_active(&rq->fair_server))
			dl_server_update(&rq->fair_server, delta_exec);
	}

	account_cfs_rq_runtime(cfs_rq, delta_exec);

	if (cfs_rq->nr_queued == 1)
		return;

	/*
	 * 这里只设置重调度请求，
	 * 实际上下文切换由调度器核心在合适时机完成。
	 */
	if (resched || !protect_slice(curr)) {
		resched_curr_lazy(rq);
		clear_buddies(cfs_rq, curr);
	}
}

/*
 * update_curr_fair - 从调度类回调进入资源提供者所在队列的运行时间更新。
 */
static void update_curr_fair(struct rq *rq)
{
	update_curr(cfs_rq_of(&rq->donor->se));
}

/*
 * update_stats_wait_start_fair：
 * 启用 schedstat 时记录实体开始在运行队列中等待的时刻。
 */
static inline void
update_stats_wait_start_fair(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	struct sched_statistics *stats;
	struct task_struct *p = NULL;

	if (!schedstat_enabled())
		return;

	stats = __schedstats_from_se(se);

	if (entity_is_task(se))
		p = task_of(se);

	__update_stats_wait_start(rq_of(cfs_rq), p, stats);
}

/*
 * update_stats_wait_end_fair - 结算等待时间，
 * 并跳过运行中才启用统计造成的无效起点。
 */
static inline void
update_stats_wait_end_fair(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	struct sched_statistics *stats;
	struct task_struct *p = NULL;

	if (!schedstat_enabled())
		return;

	stats = __schedstats_from_se(se);

	/*
	 * When the sched_schedstat changes from 0 to 1, some sched se
	 * maybe already in the runqueue, the se->statistics.wait_start
	 * will be 0.So it will let the delta wrong. We need to avoid this
	 * scenario.
	 */
	if (unlikely(!schedstat_val(stats->wait_start)))
		return;

	if (entity_is_task(se))
		p = task_of(se);

	__update_stats_wait_end(rq_of(cfs_rq), p, stats);
}

/*
 * update_stats_enqueue_sleeper_fair - 在睡眠实体唤醒入队时结算睡眠或阻塞统计。
 */
static inline void
update_stats_enqueue_sleeper_fair(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	struct sched_statistics *stats;
	struct task_struct *tsk = NULL;

	if (!schedstat_enabled())
		return;

	stats = __schedstats_from_se(se);

	if (entity_is_task(se))
		tsk = task_of(se);

	__update_stats_enqueue_sleeper(rq_of(cfs_rq), tsk, stats);
}

/*
 * Task is being enqueued - update stats:
 */
/*
 * update_stats_enqueue_fair - 按入队原因开始等待计时，
 * 并在唤醒时结束睡眠计时。
 */
static inline void
update_stats_enqueue_fair(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{
	if (!schedstat_enabled())
		return;

	/*
	 * Are we enqueueing a waiting task? (for current tasks
	 * a dequeue/enqueue event is a NOP)
	 */
	if (se != cfs_rq->curr)
		update_stats_wait_start_fair(cfs_rq, se);

	if (flags & ENQUEUE_WAKEUP)
		update_stats_enqueue_sleeper_fair(cfs_rq, se);
}

/*
 * update_stats_dequeue_fair - 结束排队等待计时，
 * 并按任务状态记录睡眠或阻塞起点。
 */
static inline void
update_stats_dequeue_fair(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{

	if (!schedstat_enabled())
		return;

	/*
	 * Mark the end of the wait period if dequeueing a
	 * waiting task:
	 */
	if (se != cfs_rq->curr)
		update_stats_wait_end_fair(cfs_rq, se);

	if ((flags & DEQUEUE_SLEEP) && entity_is_task(se)) {
		struct task_struct *tsk = task_of(se);
		unsigned int state;

		/* XXX racy against TTWU */
		state = READ_ONCE(tsk->__state);
		if (state & TASK_INTERRUPTIBLE)
			__schedstat_set(tsk->stats.sleep_start,
				      rq_clock(rq_of(cfs_rq)));
		if (state & TASK_UNINTERRUPTIBLE)
			__schedstat_set(tsk->stats.block_start,
				      rq_clock(rq_of(cfs_rq)));
	}
}

/*
 * We are picking a new current task - update its stats:
 */
/*
 * update_stats_curr_start - 记录实体本轮执行的起始任务时钟。
 */
static inline void
update_stats_curr_start(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	/*
	 * We are starting a new run period:
	 */
	se->exec_start = rq_clock_task(rq_of(cfs_rq));
}

/**************************************************
 * Scheduling class queueing methods:
 */

/*
 * is_core_idle - 检查该 CPU 的其他 SMT 兄弟是否空闲。
 * 调用处还需保证目标 CPU 自身满足空闲条件。
 */
static inline bool is_core_idle(int cpu)
{
#ifdef CONFIG_SCHED_SMT
	int sibling;

	for_each_cpu(sibling, cpu_smt_mask(cpu)) {
		if (cpu == sibling)
			continue;

		if (!idle_cpu(sibling))
			return false;
	}
#endif

	return true;
}

#ifdef CONFIG_NUMA
#define NUMA_IMBALANCE_MIN 2

/*
 * adjust_numa_imbalance - 在目标节点任务较少时容忍小幅不均衡，
 * 减少无益的跨节点搬迁。
 */
static inline long
adjust_numa_imbalance(int imbalance, int dst_running, int imb_numa_nr)
{
	/*
	 * Allow a NUMA imbalance if busy CPUs is less than the maximum
	 * threshold. Above this threshold, individual tasks may be contending
	 * for both memory bandwidth and any shared HT resources.  This is an
	 * approximation as the number of running tasks may not be related to
	 * the number of busy CPUs due to sched_setaffinity.
	 */
	if (dst_running > imb_numa_nr)
		return imbalance;

	/*
	 * Allow a small imbalance based on a simple pair of communicating
	 * tasks that remain local when the destination is lightly loaded.
	 */
	if (imbalance <= NUMA_IMBALANCE_MIN)
		return 0;

	return imbalance;
}
#endif /* CONFIG_NUMA */

#ifdef CONFIG_NUMA_BALANCING
/*
 * Approximate time to scan a full NUMA task in ms. The task scan period is
 * calculated based on the tasks virtual memory size and
 * numa_balancing_scan_size.
 */
unsigned int sysctl_numa_balancing_scan_period_min = 1000;
unsigned int sysctl_numa_balancing_scan_period_max = 60000;

/* Portion of address space to scan in MB */
unsigned int sysctl_numa_balancing_scan_size = 256;

/* Scan @scan_size MB every @scan_period after an initial @scan_delay in ms */
unsigned int sysctl_numa_balancing_scan_delay = 1000;

/* The page with hint page fault latency < threshold in ms is considered hot */
unsigned int sysctl_numa_balancing_hot_threshold = MSEC_PER_SEC;

/*
 * NUMA 组把检测到共享内存访问的任务聚合起来，
 * 以共同访问分布指导放置。
 * 它与控制 CPU 份额的 task_group/cgroup 是不同的数据结构。
 */
struct numa_group {
	refcount_t refcount;

	spinlock_t lock; /* nr_tasks, tasks */
	int nr_tasks;
	pid_t gid;
	int active_nodes;

	struct rcu_head rcu;
	unsigned long total_faults;
	unsigned long max_faults_cpu;
	/*
	 * faults[] array is split into two regions: faults_mem and faults_cpu.
	 *
	 * Faults_cpu is used to decide whether memory should move
	 * towards the CPU. As a consequence, these stats are weighted
	 * more by CPU use than by memory faults.
	 */
	unsigned long faults[];
};

/*
 * For functions that can be called in multiple contexts that permit reading
 * ->numa_group (see struct task_struct for locking rules).
 */
/*
 * deref_task_numa_group：
 * 按当前任务或已锁定且未运行任务的条件读取 NUMA 组指针。
 */
static struct numa_group *deref_task_numa_group(struct task_struct *p)
{
	return rcu_dereference_check(p->numa_group, p == current ||
		(lockdep_is_held(__rq_lockp(task_rq(p))) && !READ_ONCE(p->on_cpu)));
}

/*
 * deref_curr_numa_group - 读取当前任务的 NUMA 组，
 * 利用当前任务身份保证访问条件。
 */
static struct numa_group *deref_curr_numa_group(struct task_struct *p)
{
	return rcu_dereference_protected(p->numa_group, p == current);
}

static inline unsigned long group_faults_priv(struct numa_group *ng);
static inline unsigned long group_faults_shared(struct numa_group *ng);

/*
 * task_nr_scan_windows：
 * 按常驻内存大小和单次扫描量估算完整扫描所需的窗口数。
 */
static unsigned int task_nr_scan_windows(struct task_struct *p)
{
	unsigned long rss = 0;
	unsigned long nr_scan_pages;

	/*
	 * Calculations based on RSS as non-present and empty pages are skipped
	 * by the PTE scanner and NUMA hinting faults should be trapped based
	 * on resident pages
	 */
	nr_scan_pages = MB_TO_PAGES(sysctl_numa_balancing_scan_size);
	rss = get_mm_rss(p->mm);
	if (!rss)
		rss = nr_scan_pages;

	rss = round_up(rss, nr_scan_pages);
	return rss / nr_scan_pages;
}

/* For sanity's sake, never scan more PTEs than MAX_SCAN_WINDOW MB/sec. */
#define MAX_SCAN_WINDOW 2560

/*
 * task_scan_min - 计算单次 NUMA 扫描的最短间隔，并限制过高的扫描速率。
 */
static unsigned int task_scan_min(struct task_struct *p)
{
	unsigned int scan_size = READ_ONCE(sysctl_numa_balancing_scan_size);
	unsigned int scan, floor;
	unsigned int windows = 1;

	if (scan_size < MAX_SCAN_WINDOW)
		windows = MAX_SCAN_WINDOW / scan_size;
	floor = 1000 / windows;

	scan = sysctl_numa_balancing_scan_period_min / task_nr_scan_windows(p);
	return max_t(unsigned int, floor, scan);
}

/*
 * task_scan_start：
 * 根据扫描下限、NUMA 组规模和共享访问比例确定初始扫描间隔。
 */
static unsigned int task_scan_start(struct task_struct *p)
{
	unsigned long smin = task_scan_min(p);
	unsigned long period = smin;
	struct numa_group *ng;

	/* Scale the maximum scan period with the amount of shared memory. */
	rcu_read_lock();
	ng = rcu_dereference(p->numa_group);
	if (ng) {
		unsigned long shared = group_faults_shared(ng);
		unsigned long private = group_faults_priv(ng);

		period *= refcount_read(&ng->refcount);
		period *= shared + 1;
		period /= private + shared + 1;
	}
	rcu_read_unlock();

	return max(smin, period);
}

/*
 * task_scan_max - 计算扫描间隔上限，并为大量共享内存的 NUMA 组放宽该上限。
 */
static unsigned int task_scan_max(struct task_struct *p)
{
	unsigned long smin = task_scan_min(p);
	unsigned long smax;
	struct numa_group *ng;

	/* Watch for min being lower than max due to floor calculations */
	smax = sysctl_numa_balancing_scan_period_max / task_nr_scan_windows(p);

	/* Scale the maximum scan period with the amount of shared memory. */
	ng = deref_curr_numa_group(p);
	if (ng) {
		unsigned long shared = group_faults_shared(ng);
		unsigned long private = group_faults_priv(ng);
		unsigned long period = smax;

		period *= refcount_read(&ng->refcount);
		period *= shared + 1;
		period /= private + shared + 1;

		smax = max(smax, period);
	}

	return max(smin, smax);
}

/*
 * account_numa_enqueue - 累计有首选节点的任务数，
 * 以及当前位于首选节点的任务数。
 */
static void account_numa_enqueue(struct rq *rq, struct task_struct *p)
{
	rq->nr_numa_running += (p->numa_preferred_nid != NUMA_NO_NODE);
	rq->nr_preferred_running += (p->numa_preferred_nid == task_node(p));
}

/*
 * account_numa_dequeue - 撤销任务对运行队列 NUMA 分布计数的贡献。
 */
static void account_numa_dequeue(struct rq *rq, struct task_struct *p)
{
	rq->nr_numa_running -= (p->numa_preferred_nid != NUMA_NO_NODE);
	rq->nr_preferred_running -= (p->numa_preferred_nid == task_node(p));
}

/* Shared or private faults. */
#define NR_NUMA_HINT_FAULT_TYPES 2

/* Memory and CPU locality */
#define NR_NUMA_HINT_FAULT_STATS (NR_NUMA_HINT_FAULT_TYPES * 2)

/* Averaged statistics, and temporary buffers. */
#define NR_NUMA_HINT_FAULT_BUCKETS (NR_NUMA_HINT_FAULT_STATS * 2)

/*
 * task_numa_group_id - 在 RCU 保护下读取任务所属 NUMA 组编号，无组时返回 0。
 */
pid_t task_numa_group_id(struct task_struct *p)
{
	struct numa_group *ng;
	pid_t gid = 0;

	rcu_read_lock();
	ng = rcu_dereference(p->numa_group);
	if (ng)
		gid = ng->gid;
	rcu_read_unlock();

	return gid;
}

/*
 * The averaged statistics, shared & private, memory & CPU,
 * occupy the first half of the array. The second half of the
 * array is for current counters, which are averaged into the
 * first set by task_numa_placement.
 */
/*
 * task_faults_idx：
 * 把统计类别、节点编号和私有/共享属性映射为故障统计数组下标。
 */
static inline int task_faults_idx(enum numa_faults_stats s, int nid, int priv)
{
	return NR_NUMA_HINT_FAULT_TYPES * (s * nr_node_ids + nid) + priv;
}

/*
 * task_faults - 合计任务访问指定内存节点的私有和共享 NUMA 提示故障统计。
 */
static inline unsigned long task_faults(struct task_struct *p, int nid)
{
	if (!p->numa_faults)
		return 0;

	return p->numa_faults[task_faults_idx(NUMA_MEM, nid, 0)] +
		p->numa_faults[task_faults_idx(NUMA_MEM, nid, 1)];
}

/*
 * group_faults - 合计任务所属 NUMA 组对指定内存节点的访问统计。
 */
static inline unsigned long group_faults(struct task_struct *p, int nid)
{
	struct numa_group *ng = deref_task_numa_group(p);

	if (!ng)
		return 0;

	return ng->faults[task_faults_idx(NUMA_MEM, nid, 0)] +
		ng->faults[task_faults_idx(NUMA_MEM, nid, 1)];
}

/*
 * group_faults_cpu - 合计 NUMA 组在指定 CPU 节点上产生的提示故障统计。
 */
static inline unsigned long group_faults_cpu(struct numa_group *group, int nid)
{
	return group->faults[task_faults_idx(NUMA_CPU, nid, 0)] +
		group->faults[task_faults_idx(NUMA_CPU, nid, 1)];
}

/*
 * group_faults_priv - 汇总 NUMA 组在各内存节点上的私有访问统计。
 */
static inline unsigned long group_faults_priv(struct numa_group *ng)
{
	unsigned long faults = 0;
	int node;

	for_each_online_node(node) {
		faults += ng->faults[task_faults_idx(NUMA_MEM, node, 1)];
	}

	return faults;
}

/*
 * group_faults_shared - 汇总 NUMA 组在各内存节点上的共享访问统计。
 */
static inline unsigned long group_faults_shared(struct numa_group *ng)
{
	unsigned long faults = 0;
	int node;

	for_each_online_node(node) {
		faults += ng->faults[task_faults_idx(NUMA_MEM, node, 0)];
	}

	return faults;
}

/*
 * A node triggering more than 1/3 as many NUMA faults as the maximum is
 * considered part of a numa group's pseudo-interleaving set. Migrations
 * between these nodes are slowed down, to allow things to settle down.
 */
#define ACTIVE_NODE_FRACTION 3

/*
 * numa_is_active_node - 以组内最活跃节点为参照，
 * 判断节点是否承担了足够多的访问。
 */
static bool numa_is_active_node(int nid, struct numa_group *ng)
{
	return group_faults_cpu(ng, nid) * ACTIVE_NODE_FRACTION > ng->max_faults_cpu;
}

/* Handle placement on systems where not all nodes are directly connected. */
/*
 * score_nearby_nodes - 结合 NUMA 拓扑距离，
 * 把邻近内存节点的访问计入候选节点得分。
 */
static unsigned long score_nearby_nodes(struct task_struct *p, int nid,
					int lim_dist, bool task)
{
	unsigned long score = 0;
	int node, max_dist;

	/*
	 * All nodes are directly connected, and the same distance
	 * from each other. No need for fancy placement algorithms.
	 */
	if (sched_numa_topology_type == NUMA_DIRECT)
		return 0;

	/* sched_max_numa_distance may be changed in parallel. */
	max_dist = READ_ONCE(sched_max_numa_distance);
	/*
	 * This code is called for each node, introducing N^2 complexity,
	 * which should be OK given the number of nodes rarely exceeds 8.
	 */
	for_each_online_node(node) {
		unsigned long faults;
		int dist = node_distance(nid, node);

		/*
		 * The furthest away nodes in the system are not interesting
		 * for placement; nid was already counted.
		 */
		if (dist >= max_dist || node == nid)
			continue;

		/*
		 * On systems with a backplane NUMA topology, compare groups
		 * of nodes, and move tasks towards the group with the most
		 * memory accesses. When comparing two nodes at distance
		 * "hoplimit", only nodes closer by than "hoplimit" are part
		 * of each group. Skip other nodes.
		 */
		if (sched_numa_topology_type == NUMA_BACKPLANE && dist >= lim_dist)
			continue;

		/* Add up the faults from nearby nodes. */
		if (task)
			faults = task_faults(p, node);
		else
			faults = group_faults(p, node);

		/*
		 * On systems with a glueless mesh NUMA topology, there are
		 * no fixed "groups of nodes". Instead, nodes that are not
		 * directly connected bounce traffic through intermediate
		 * nodes; a numa_group can occupy any set of nodes.
		 * The further away a node is, the less the faults count.
		 * This seems to result in good task placement.
		 */
		if (sched_numa_topology_type == NUMA_GLUELESS_MESH) {
			faults *= (max_dist - dist);
			faults /= (max_dist - LOCAL_DISTANCE);
		}

		score += faults;
	}

	return score;
}

/*
 * These return the fraction of accesses done by a particular task, or
 * task group, on a particular numa node.  The group weight is given a
 * larger multiplier, in order to group tasks together that are almost
 * evenly spread out between numa nodes.
 */
/*
 * task_weight - 把任务对节点及邻近节点的访问偏好归一化到千分比尺度。
 */
static inline unsigned long task_weight(struct task_struct *p, int nid,
					int dist)
{
	unsigned long faults, total_faults;

	if (!p->numa_faults)
		return 0;

	total_faults = p->total_numa_faults;

	if (!total_faults)
		return 0;

	faults = task_faults(p, nid);
	faults += score_nearby_nodes(p, nid, dist, true);

	return 1000 * faults / total_faults;
}

/*
 * group_weight - 按 NUMA 组的总访问量归一化候选节点的局部性得分。
 */
static inline unsigned long group_weight(struct task_struct *p, int nid,
					 int dist)
{
	struct numa_group *ng = deref_task_numa_group(p);
	unsigned long faults, total_faults;

	if (!ng)
		return 0;

	total_faults = ng->total_faults;

	if (!total_faults)
		return 0;

	faults = group_faults(p, nid);
	faults += score_nearby_nodes(p, nid, dist, false);

	return 1000 * faults / total_faults;
}

/*
 * If memory tiering mode is enabled, cpupid of slow memory page is
 * used to record scan time instead of CPU and PID.  When tiering mode
 * is disabled at run time, the scan time (in cpupid) will be
 * interpreted as CPU and PID.  So CPU needs to be checked to avoid to
 * access out of array bound.
 */
/*
 * cpupid_valid - 检查页上记录的 CPU/PID 信息中，CPU 编号是否处于有效范围。
 */
static inline bool cpupid_valid(int cpupid)
{
	return cpupid_to_cpu(cpupid) < nr_cpu_ids;
}

/*
 * For memory tiering mode, if there are enough free pages (more than
 * enough watermark defined here) in fast memory node, to take full
 * advantage of fast memory capacity, all recently accessed slow
 * memory pages will be migrated to fast memory node without
 * considering hot threshold.
 */
/*
 * pgdat_free_space_enough：
 * 检查目标内存节点是否有高于提升水位的充足空闲空间。
 */
static bool pgdat_free_space_enough(struct pglist_data *pgdat)
{
	int z;
	unsigned long enough_wmark;

	enough_wmark = max(1UL * 1024 * 1024 * 1024 >> PAGE_SHIFT,
			   pgdat->node_present_pages >> 4);
	for (z = pgdat->nr_zones - 1; z >= 0; z--) {
		struct zone *zone = pgdat->node_zones + z;

		if (!populated_zone(zone))
			continue;

		if (zone_watermark_ok(zone, 0,
				      promo_wmark_pages(zone) + enough_wmark,
				      ZONE_MOVABLE, 0))
			return true;
	}
	return false;
}

/*
 * For memory tiering mode, when page tables are scanned, the scan
 * time will be recorded in struct page in addition to make page
 * PROT_NONE for slow memory page.  So when the page is accessed, in
 * hint page fault handler, the hint page fault latency is calculated
 * via,
 *
 *	hint page fault latency = hint page fault time - scan time
 *
 * The smaller the hint page fault latency, the higher the possibility
 * for the page to be hot.
 */
/*
 * numa_hint_fault_latency - 读取并更新页访问时间，
 * 计算用于判断冷热的提示故障延迟。
 */
static int numa_hint_fault_latency(struct folio *folio)
{
	int last_time, time;

	time = jiffies_to_msecs(jiffies);
	last_time = folio_xchg_access_time(folio, time);

	return (time - last_time) & PAGE_ACCESS_TIME_MASK;
}

/*
 * For memory tiering mode, too high promotion/demotion throughput may
 * hurt application latency.  So we provide a mechanism to rate limit
 * the number of pages that are tried to be promoted.
 */
/*
 * numa_promotion_rate_limit - 累计候选提升页数，
 * 达到节点每秒提升预算时返回 true。
 */
static bool numa_promotion_rate_limit(struct pglist_data *pgdat,
				      unsigned long rate_limit, int nr)
{
	unsigned long nr_cand;
	unsigned int now, start;

	now = jiffies_to_msecs(jiffies);
	mod_node_page_state(pgdat, PGPROMOTE_CANDIDATE, nr);
	nr_cand = node_page_state(pgdat, PGPROMOTE_CANDIDATE);
	start = pgdat->nbp_rl_start;
	if (now - start > MSEC_PER_SEC &&
	    cmpxchg(&pgdat->nbp_rl_start, start, now) == start)
		pgdat->nbp_rl_nr_cand = nr_cand;
	if (nr_cand - pgdat->nbp_rl_nr_cand >= rate_limit)
		return true;
	return false;
}

#define NUMA_MIGRATION_ADJUST_STEPS	16

/*
 * numa_promotion_adjust_threshold - 根据候选提升速率反馈调整热页判定阈值。
 */
static void numa_promotion_adjust_threshold(struct pglist_data *pgdat,
					    unsigned long rate_limit,
					    unsigned int ref_th)
{
	unsigned int now, start, th_period, unit_th, th;
	unsigned long nr_cand, ref_cand, diff_cand;

	now = jiffies_to_msecs(jiffies);
	th_period = sysctl_numa_balancing_scan_period_max;
	start = pgdat->nbp_th_start;
	if (now - start > th_period &&
	    cmpxchg(&pgdat->nbp_th_start, start, now) == start) {
		ref_cand = rate_limit *
			sysctl_numa_balancing_scan_period_max / MSEC_PER_SEC;
		nr_cand = node_page_state(pgdat, PGPROMOTE_CANDIDATE);
		diff_cand = nr_cand - pgdat->nbp_th_nr_cand;
		unit_th = ref_th * 2 / NUMA_MIGRATION_ADJUST_STEPS;
		th = pgdat->nbp_threshold ? : ref_th;
		if (diff_cand > ref_cand * 11 / 10)
			th = max(th - unit_th, unit_th);
		else if (diff_cand < ref_cand * 9 / 10)
			th = min(th + unit_th, ref_th * 2);
		pgdat->nbp_th_nr_cand = nr_cand;
		pgdat->nbp_threshold = th;
	}
}

/*
 * should_numa_migrate_memory - 决定是否将内存页迁到目标 CPU 所在节点。
 * 分层内存考虑热度和提升速率，
 * 普通 NUMA 迁移考虑访问历史及共享局部性。
 */
bool should_numa_migrate_memory(struct task_struct *p, struct folio *folio,
				int src_nid, int dst_cpu)
{
	struct numa_group *ng = deref_curr_numa_group(p);
	int dst_nid = cpu_to_node(dst_cpu);
	int last_cpupid, this_cpupid;

	/*
	 * Cannot migrate to memoryless nodes.
	 */
	if (!node_state(dst_nid, N_MEMORY))
		return false;

	/*
	 * The pages in slow memory node should be migrated according
	 * to hot/cold instead of private/shared.
	 */
	/*
	 * 分层内存场景用访问延迟估计热度；
	 * 目标空间宽裕时可跳过热度和速率筛选。
	 */
	if (folio_use_access_time(folio)) {
		struct pglist_data *pgdat;
		unsigned long rate_limit;
		unsigned int latency, th, def_th;
		long nr = folio_nr_pages(folio);

		pgdat = NODE_DATA(dst_nid);
		if (pgdat_free_space_enough(pgdat)) {
			/* workload changed, reset hot threshold */
			pgdat->nbp_threshold = 0;
			mod_node_page_state(pgdat, PGPROMOTE_CANDIDATE_NRL, nr);
			return true;
		}

		def_th = sysctl_numa_balancing_hot_threshold;
		rate_limit = MB_TO_PAGES(sysctl_numa_balancing_promote_rate_limit);
		numa_promotion_adjust_threshold(pgdat, rate_limit, def_th);

		th = pgdat->nbp_threshold ? : def_th;
		latency = numa_hint_fault_latency(folio);
		if (latency >= th)
			return false;

		return !numa_promotion_rate_limit(pgdat, rate_limit, nr);
	}

	this_cpupid = cpu_pid_to_cpupid(dst_cpu, current->pid);
	last_cpupid = folio_xchg_last_cpupid(folio, this_cpupid);

	if (!(sysctl_numa_balancing_mode & NUMA_BALANCING_MEMORY_TIERING) &&
	    !node_is_toptier(src_nid) && !cpupid_valid(last_cpupid))
		return false;

	/*
	 * Allow first faults or private faults to migrate immediately early in
	 * the lifetime of a task. The magic number 4 is based on waiting for
	 * two full passes of the "multi-stage node selection" test that is
	 * executed below.
	 */
	if ((p->numa_preferred_nid == NUMA_NO_NODE || p->numa_scan_seq <= 4) &&
	    (cpupid_pid_unset(last_cpupid) || cpupid_match_pid(p, last_cpupid)))
		return true;

	/*
	 * Multi-stage node selection is used in conjunction with a periodic
	 * migration fault to build a temporal task<->page relation. By using
	 * a two-stage filter we remove short/unlikely relations.
	 *
	 * Using P(p) ~ n_p / n_t as per frequentist probability, we can equate
	 * a task's usage of a particular page (n_p) per total usage of this
	 * page (n_t) (in a given time-span) to a probability.
	 *
	 * Our periodic faults will sample this probability and getting the
	 * same result twice in a row, given these samples are fully
	 * independent, is then given by P(n)^2, provided our sample period
	 * is sufficiently short compared to the usage pattern.
	 *
	 * This quadric squishes small probabilities, making it less likely we
	 * act on an unlikely task<->page relation.
	 */
	if (!cpupid_pid_unset(last_cpupid) &&
				cpupid_to_nid(last_cpupid) != dst_nid)
		return false;

	/* Always allow migrate on private faults */
	if (cpupid_match_pid(p, last_cpupid))
		return true;

	/* A shared fault, but p->numa_group has not been set up yet. */
	if (!ng)
		return true;

	/*
	 * Destination node is much more heavily used than the source
	 * node? Allow migration.
	 */
	if (group_faults_cpu(ng, dst_nid) > group_faults_cpu(ng, src_nid) *
					ACTIVE_NODE_FRACTION)
		return true;

	/*
	 * Distribute memory according to CPU & memory use on each node,
	 * with 3/4 hysteresis to avoid unnecessary memory migrations:
	 *
	 * faults_cpu(dst)   3   faults_cpu(src)
	 * --------------- * - > ---------------
	 * faults_mem(dst)   4   faults_mem(src)
	 */
	return group_faults_cpu(ng, dst_nid) * group_faults(p, src_nid) * 3 >
	       group_faults_cpu(ng, src_nid) * group_faults(p, dst_nid) * 4;
}

/*
 * 'numa_type' describes the node at the moment of load balancing.
 */
enum numa_type {
	/* The node has spare capacity that can be used to run more tasks.  */
	node_has_spare = 0,
	/*
	 * The node is fully used and the tasks don't compete for more CPU
	 * cycles. Nevertheless, some tasks might wait before running.
	 */
	node_fully_busy,
	/*
	 * The node is overloaded and can't provide expected CPU cycles to all
	 * tasks.
	 */
	node_overloaded
};

/* Cached statistics for all CPUs within a node */
struct numa_stats {
	unsigned long load;
	unsigned long runnable;
	unsigned long util;
	/* Total compute capacity of CPUs on a node */
	unsigned long compute_capacity;
	unsigned int nr_running;
	unsigned int weight;
	enum numa_type node_type;
	int idle_cpu;
};

struct task_numa_env {
	struct task_struct *p;

	int src_cpu, src_nid;
	int dst_cpu, dst_nid;
	int imb_numa_nr;

	struct numa_stats src_stats, dst_stats;

	int imbalance_pct;
	int dist;

	struct task_struct *best_task;
	long best_imp;
	int best_cpu;
};

static unsigned long cpu_load(struct rq *rq);
static unsigned long cpu_runnable(struct rq *rq);

/*
 * numa_classify：
 * 按任务数、利用率和容量将 NUMA 节点分为有余量、全忙或过载。
 */
static inline enum
numa_type numa_classify(unsigned int imbalance_pct,
			 struct numa_stats *ns)
{
	if ((ns->nr_running > ns->weight) &&
	    (((ns->compute_capacity * 100) < (ns->util * imbalance_pct)) ||
	     ((ns->compute_capacity * imbalance_pct) < (ns->runnable * 100))))
		return node_overloaded;

	if ((ns->nr_running < ns->weight) ||
	    (((ns->compute_capacity * 100) > (ns->util * imbalance_pct)) &&
	     ((ns->compute_capacity * imbalance_pct) > (ns->runnable * 100))))
		return node_has_spare;

	return node_fully_busy;
}

#ifdef CONFIG_SCHED_SMT
/* Forward declarations of select_idle_sibling helpers */
static inline bool test_idle_cores(int cpu);
/*
 * numa_idle_core - 在尚未找到空闲核心时，
 * 检查候选 CPU 的 SMT 兄弟是否也空闲。
 */
static inline int numa_idle_core(int idle_core, int cpu)
{
	if (!static_branch_likely(&sched_smt_present) ||
	    idle_core >= 0 || !test_idle_cores(cpu))
		return idle_core;

	/*
	 * Prefer cores instead of packing HT siblings
	 * and triggering future load balancing.
	 */
	if (is_core_idle(cpu))
		idle_core = cpu;

	return idle_core;
}
#else /* !CONFIG_SCHED_SMT: */
/*
 * numa_idle_core - 未启用 SMT 时，无需额外执行整核空闲检查。
 */
static inline int numa_idle_core(int idle_core, int cpu)
{
	return idle_core;
}
#endif /* !CONFIG_SCHED_SMT */

/*
 * Gather all necessary information to make NUMA balancing placement
 * decisions that are compatible with standard load balancer. This
 * borrows code and logic from update_sg_lb_stats but sharing a
 * common implementation is impractical.
 */
/*
 * update_numa_stats - 汇总目标节点的负载、利用率和容量，
 * 并按需寻找可用空闲 CPU。
 */
static void update_numa_stats(struct task_numa_env *env,
			      struct numa_stats *ns, int nid,
			      bool find_idle)
{
	int cpu, idle_core = -1;

	memset(ns, 0, sizeof(*ns));
	ns->idle_cpu = -1;

	rcu_read_lock();
	for_each_cpu(cpu, cpumask_of_node(nid)) {
		struct rq *rq = cpu_rq(cpu);

		ns->load += cpu_load(rq);
		ns->runnable += cpu_runnable(rq);
		ns->util += cpu_util_cfs(cpu);
		ns->nr_running += rq->cfs.h_nr_runnable;
		ns->compute_capacity += capacity_of(cpu);

		if (find_idle && idle_core < 0 && !rq->nr_running && idle_cpu(cpu)) {
			if (READ_ONCE(rq->numa_migrate_on) ||
			    !cpumask_test_cpu(cpu, env->p->cpus_ptr))
				continue;

			if (ns->idle_cpu == -1)
				ns->idle_cpu = cpu;

			idle_core = numa_idle_core(idle_core, cpu);
		}
	}
	rcu_read_unlock();

	ns->weight = cpumask_weight(cpumask_of_node(nid));

	ns->node_type = numa_classify(env->imbalance_pct, ns);

	if (idle_core >= 0)
		ns->idle_cpu = idle_core;
}

/*
 * task_numa_assign - 保存当前最优迁移或交换候选，
 * 并占用目标 CPU 的 NUMA 迁移标记。
 */
static void task_numa_assign(struct task_numa_env *env,
			     struct task_struct *p, long imp)
{
	struct rq *rq = cpu_rq(env->dst_cpu);

	/* Check if run-queue part of active NUMA balance. */
	if (env->best_cpu != env->dst_cpu && xchg(&rq->numa_migrate_on, 1)) {
		int cpu;
		int start = env->dst_cpu;

		/* Find alternative idle CPU. */
		for_each_cpu_wrap(cpu, cpumask_of_node(env->dst_nid), start + 1) {
			if (cpu == env->best_cpu || !idle_cpu(cpu) ||
			    !cpumask_test_cpu(cpu, env->p->cpus_ptr)) {
				continue;
			}

			env->dst_cpu = cpu;
			rq = cpu_rq(env->dst_cpu);
			if (!xchg(&rq->numa_migrate_on, 1))
				goto assign;
		}

		/* Failed to find an alternative idle CPU */
		return;
	}

assign:
	/*
	 * Clear previous best_cpu/rq numa-migrate flag, since task now
	 * found a better CPU to move/swap.
	 */
	if (env->best_cpu != -1 && env->best_cpu != env->dst_cpu) {
		rq = cpu_rq(env->best_cpu);
		WRITE_ONCE(rq->numa_migrate_on, 0);
	}

	if (env->best_task)
		put_task_struct(env->best_task);
	if (p)
		get_task_struct(p);

	env->best_task = p;
	env->best_imp = imp;
	env->best_cpu = env->dst_cpu;
}

/*
 * load_too_imbalanced - 按两端容量归一化负载，
 * 判断迁移后的不均衡是否比原来更严重。
 */
static bool load_too_imbalanced(long src_load, long dst_load,
				struct task_numa_env *env)
{
	long imb, old_imb;
	long orig_src_load, orig_dst_load;
	long src_capacity, dst_capacity;

	/*
	 * The load is corrected for the CPU capacity available on each node.
	 *
	 * src_load        dst_load
	 * ------------ vs ---------
	 * src_capacity    dst_capacity
	 */
	src_capacity = env->src_stats.compute_capacity;
	dst_capacity = env->dst_stats.compute_capacity;

	imb = abs(dst_load * src_capacity - src_load * dst_capacity);

	orig_src_load = env->src_stats.load;
	orig_dst_load = env->dst_stats.load;

	old_imb = abs(orig_dst_load * src_capacity - orig_src_load * dst_capacity);

	/* Would this change make things worse? */
	return (imb > old_imb);
}

/*
 * Maximum NUMA importance can be 1998 (2*999);
 * SMALLIMP @ 30 would be close to 1998/64.
 * Used to deter task migration.
 */
#define SMALLIMP	30

/*
 * This checks if the overall compute and NUMA accesses of the system would
 * be improved if the source tasks was migrated to the target dst_cpu taking
 * into account that it might be best if task running on the dst_cpu should
 * be exchanged with the source task
 */
/*
 * task_numa_compare - 评估与目标 CPU 当前任务交换或直接迁移的局部性收益。
 * 同时遵守亲和性及负载约束；返回 true 表示可提前结束候选搜索。
 */
static bool task_numa_compare(struct task_numa_env *env,
			      long taskimp, long groupimp, bool maymove)
{
	struct numa_group *cur_ng, *p_ng = deref_curr_numa_group(env->p);
	struct rq *dst_rq = cpu_rq(env->dst_cpu);
	long imp = p_ng ? groupimp : taskimp;
	struct task_struct *cur;
	long src_load, dst_load;
	int dist = env->dist;
	long moveimp = imp;
	long load;
	bool stopsearch = false;

	if (READ_ONCE(dst_rq->numa_migrate_on))
		return false;

	rcu_read_lock();
	cur = rcu_dereference(dst_rq->curr);
	if (cur && ((cur->flags & (PF_EXITING | PF_KTHREAD)) ||
		    !cur->mm))
		cur = NULL;

	/*
	 * Because we have preemption enabled we can get migrated around and
	 * end try selecting ourselves (current == env->p) as a swap candidate.
	 */
	if (cur == env->p) {
		stopsearch = true;
		goto unlock;
	}

	if (!cur) {
		if (maymove && moveimp >= env->best_imp)
			goto assign;
		else
			goto unlock;
	}

	/* Skip this swap candidate if cannot move to the source cpu. */
	if (!cpumask_test_cpu(env->src_cpu, cur->cpus_ptr))
		goto unlock;

	/*
	 * Skip this swap candidate if it is not moving to its preferred
	 * node and the best task is.
	 */
	if (env->best_task &&
	    env->best_task->numa_preferred_nid == env->src_nid &&
	    cur->numa_preferred_nid != env->src_nid) {
		goto unlock;
	}

	/*
	 * "imp" is the fault differential for the source task between the
	 * source and destination node. Calculate the total differential for
	 * the source task and potential destination task. The more negative
	 * the value is, the more remote accesses that would be expected to
	 * be incurred if the tasks were swapped.
	 *
	 * If dst and source tasks are in the same NUMA group, or not
	 * in any group then look only at task weights.
	 */
	cur_ng = rcu_dereference(cur->numa_group);
	if (cur_ng == p_ng) {
		/*
		 * Do not swap within a group or between tasks that have
		 * no group if there is spare capacity. Swapping does
		 * not address the load imbalance and helps one task at
		 * the cost of punishing another.
		 */
		if (env->dst_stats.node_type == node_has_spare)
			goto unlock;

		imp = taskimp + task_weight(cur, env->src_nid, dist) -
		      task_weight(cur, env->dst_nid, dist);
		/*
		 * Add some hysteresis to prevent swapping the
		 * tasks within a group over tiny differences.
		 */
		if (cur_ng)
			imp -= imp / 16;
	} else {
		/*
		 * Compare the group weights. If a task is all by itself
		 * (not part of a group), use the task weight instead.
		 */
		if (cur_ng && p_ng)
			imp += group_weight(cur, env->src_nid, dist) -
			       group_weight(cur, env->dst_nid, dist);
		else
			imp += task_weight(cur, env->src_nid, dist) -
			       task_weight(cur, env->dst_nid, dist);
	}

	/* Discourage picking a task already on its preferred node */
	if (cur->numa_preferred_nid == env->dst_nid)
		imp -= imp / 16;

	/*
	 * Encourage picking a task that moves to its preferred node.
	 * This potentially makes imp larger than it's maximum of
	 * 1998 (see SMALLIMP and task_weight for why) but in this
	 * case, it does not matter.
	 */
	if (cur->numa_preferred_nid == env->src_nid)
		imp += imp / 8;

	if (maymove && moveimp > imp && moveimp > env->best_imp) {
		imp = moveimp;
		cur = NULL;
		goto assign;
	}

	/*
	 * Prefer swapping with a task moving to its preferred node over a
	 * task that is not.
	 */
	if (env->best_task && cur->numa_preferred_nid == env->src_nid &&
	    env->best_task->numa_preferred_nid != env->src_nid) {
		goto assign;
	}

	/*
	 * If the NUMA importance is less than SMALLIMP,
	 * task migration might only result in ping pong
	 * of tasks and also hurt performance due to cache
	 * misses.
	 */
	if (imp < SMALLIMP || imp <= env->best_imp + SMALLIMP / 2)
		goto unlock;

	/*
	 * In the overloaded case, try and keep the load balanced.
	 */
	load = task_h_load(env->p) - task_h_load(cur);
	if (!load)
		goto assign;

	dst_load = env->dst_stats.load + load;
	src_load = env->src_stats.load - load;

	if (load_too_imbalanced(src_load, dst_load, env))
		goto unlock;

assign:
	/* Evaluate an idle CPU for a task numa move. */
	if (!cur) {
		int cpu = env->dst_stats.idle_cpu;

		/* Nothing cached so current CPU went idle since the search. */
		if (cpu < 0)
			cpu = env->dst_cpu;

		/*
		 * If the CPU is no longer truly idle and the previous best CPU
		 * is, keep using it.
		 */
		if (!idle_cpu(cpu) && env->best_cpu >= 0 &&
		    idle_cpu(env->best_cpu)) {
			cpu = env->best_cpu;
		}

		env->dst_cpu = cpu;
	}

	task_numa_assign(env, cur, imp);

	/*
	 * If a move to idle is allowed because there is capacity or load
	 * balance improves then stop the search. While a better swap
	 * candidate may exist, a search is not free.
	 */
	if (maymove && !cur && env->best_cpu >= 0 && idle_cpu(env->best_cpu))
		stopsearch = true;

	/*
	 * If a swap candidate must be identified and the current best task
	 * moves its preferred node then stop the search.
	 */
	if (!maymove && env->best_task &&
	    env->best_task->numa_preferred_nid == env->src_nid) {
		stopsearch = true;
	}
unlock:
	rcu_read_unlock();

	return stopsearch;
}

/*
 * task_numa_find_cpu：
 * 在目标 NUMA 节点中寻找可直接迁入的 CPU 或合适的交换任务。
 */
static void task_numa_find_cpu(struct task_numa_env *env,
				long taskimp, long groupimp)
{
	bool maymove = false;
	int cpu;

	/*
	 * If dst node has spare capacity, then check if there is an
	 * imbalance that would be overruled by the load balancer.
	 */
	if (env->dst_stats.node_type == node_has_spare) {
		unsigned int imbalance;
		int src_running, dst_running;

		/*
		 * Would movement cause an imbalance? Note that if src has
		 * more running tasks that the imbalance is ignored as the
		 * move improves the imbalance from the perspective of the
		 * CPU load balancer.
		 * */
		src_running = env->src_stats.nr_running - 1;
		dst_running = env->dst_stats.nr_running + 1;
		imbalance = max(0, dst_running - src_running);
		imbalance = adjust_numa_imbalance(imbalance, dst_running,
						  env->imb_numa_nr);

		/* Use idle CPU if there is no imbalance */
		if (!imbalance) {
			maymove = true;
			if (env->dst_stats.idle_cpu >= 0) {
				env->dst_cpu = env->dst_stats.idle_cpu;
				task_numa_assign(env, NULL, 0);
				return;
			}
		}
	} else {
		long src_load, dst_load, load;
		/*
		 * If the improvement from just moving env->p direction is better
		 * than swapping tasks around, check if a move is possible.
		 */
		load = task_h_load(env->p);
		dst_load = env->dst_stats.load + load;
		src_load = env->src_stats.load - load;
		maymove = !load_too_imbalanced(src_load, dst_load, env);
	}

	for_each_cpu(cpu, cpumask_of_node(env->dst_nid)) {
		/* Skip this CPU if the source task cannot migrate */
		if (!cpumask_test_cpu(cpu, env->p->cpus_ptr))
			continue;

		env->dst_cpu = cpu;
		if (task_numa_compare(env, taskimp, groupimp, maymove))
			break;
	}
}

/*
 * task_numa_migrate - 跨节点搜索并执行任务迁移或双向交换，
 * 兼顾局部性和负载均衡。
 */
static int task_numa_migrate(struct task_struct *p)
{
	struct task_numa_env env = {
		.p = p,

		.src_cpu = task_cpu(p),
		.src_nid = task_node(p),

		.imbalance_pct = 112,

		.best_task = NULL,
		.best_imp = 0,
		.best_cpu = -1,
	};
	unsigned long taskweight, groupweight;
	struct sched_domain *sd;
	long taskimp, groupimp;
	struct numa_group *ng;
	struct rq *best_rq;
	int nid, ret, dist;

	/*
	 * Pick the lowest SD_NUMA domain, as that would have the smallest
	 * imbalance and would be the first to start moving tasks about.
	 *
	 * And we want to avoid any moving of tasks about, as that would create
	 * random movement of tasks -- counter the numa conditions we're trying
	 * to satisfy here.
	 */
	rcu_read_lock();
	sd = rcu_dereference(per_cpu(sd_numa, env.src_cpu));
	if (sd) {
		env.imbalance_pct = 100 + (sd->imbalance_pct - 100) / 2;
		env.imb_numa_nr = sd->imb_numa_nr;
	}
	rcu_read_unlock();

	/*
	 * Cpusets can break the scheduler domain tree into smaller
	 * balance domains, some of which do not cross NUMA boundaries.
	 * Tasks that are "trapped" in such domains cannot be migrated
	 * elsewhere, so there is no point in (re)trying.
	 */
	if (unlikely(!sd)) {
		sched_setnuma(p, task_node(p));
		return -EINVAL;
	}

	env.dst_nid = p->numa_preferred_nid;
	dist = env.dist = node_distance(env.src_nid, env.dst_nid);
	taskweight = task_weight(p, env.src_nid, dist);
	groupweight = group_weight(p, env.src_nid, dist);
	update_numa_stats(&env, &env.src_stats, env.src_nid, false);
	taskimp = task_weight(p, env.dst_nid, dist) - taskweight;
	groupimp = group_weight(p, env.dst_nid, dist) - groupweight;
	update_numa_stats(&env, &env.dst_stats, env.dst_nid, true);

	/* Try to find a spot on the preferred nid. */
	task_numa_find_cpu(&env, taskimp, groupimp);

	/*
	 * Look at other nodes in these cases:
	 * - there is no space available on the preferred_nid
	 * - the task is part of a numa_group that is interleaved across
	 *   multiple NUMA nodes; in order to better consolidate the group,
	 *   we need to check other locations.
	 */
	ng = deref_curr_numa_group(p);
	if (env.best_cpu == -1 || (ng && ng->active_nodes > 1)) {
		for_each_node_state(nid, N_CPU) {
			if (nid == env.src_nid || nid == p->numa_preferred_nid)
				continue;

			dist = node_distance(env.src_nid, env.dst_nid);
			if (sched_numa_topology_type == NUMA_BACKPLANE &&
						dist != env.dist) {
				taskweight = task_weight(p, env.src_nid, dist);
				groupweight = group_weight(p, env.src_nid, dist);
			}

			/* Only consider nodes where both task and groups benefit */
			taskimp = task_weight(p, nid, dist) - taskweight;
			groupimp = group_weight(p, nid, dist) - groupweight;
			if (taskimp < 0 && groupimp < 0)
				continue;

			env.dist = dist;
			env.dst_nid = nid;
			update_numa_stats(&env, &env.dst_stats, env.dst_nid, true);
			task_numa_find_cpu(&env, taskimp, groupimp);
		}
	}

	/*
	 * If the task is part of a workload that spans multiple NUMA nodes,
	 * and is migrating into one of the workload's active nodes, remember
	 * this node as the task's preferred numa node, so the workload can
	 * settle down.
	 * A task that migrated to a second choice node will be better off
	 * trying for a better one later. Do not set the preferred node here.
	 */
	if (ng) {
		if (env.best_cpu == -1)
			nid = env.src_nid;
		else
			nid = cpu_to_node(env.best_cpu);

		if (nid != p->numa_preferred_nid)
			sched_setnuma(p, nid);
	}

	/* No better CPU than the current one was found. */
	if (env.best_cpu == -1) {
		trace_sched_stick_numa(p, env.src_cpu, NULL, -1);
		return -EAGAIN;
	}

	best_rq = cpu_rq(env.best_cpu);
	if (env.best_task == NULL) {
		ret = migrate_task_to(p, env.best_cpu);
		WRITE_ONCE(best_rq->numa_migrate_on, 0);
		if (ret != 0)
			trace_sched_stick_numa(p, env.src_cpu, NULL, env.best_cpu);
		return ret;
	}

	/*
	 * 直接迁入可能破坏节点负载平衡，
	 * 交换双方任务可同时改善访问位置。
	 */
	ret = migrate_swap(p, env.best_task, env.best_cpu, env.src_cpu);
	WRITE_ONCE(best_rq->numa_migrate_on, 0);

	if (ret != 0)
		trace_sched_stick_numa(p, env.src_cpu, env.best_task, env.best_cpu);
	put_task_struct(env.best_task);
	return ret;
}

/* Attempt to migrate a task to a CPU on the preferred node. */
/*
 * numa_migrate_preferred - 设定下次重试时间，
 * 并尝试把偏离首选节点的任务迁回。
 */
static void numa_migrate_preferred(struct task_struct *p)
{
	unsigned long interval = HZ;

	/* This task has no NUMA fault statistics yet */
	if (unlikely(p->numa_preferred_nid == NUMA_NO_NODE || !p->numa_faults))
		return;

	/* Periodically retry migrating the task to the preferred node */
	interval = min(interval, msecs_to_jiffies(p->numa_scan_period) / 16);
	p->numa_migrate_retry = jiffies + interval;

	/* Success if task is already running on preferred CPU */
	if (task_node(p) == p->numa_preferred_nid)
		return;

	/* Otherwise, try migrate to a CPU on the preferred node */
	task_numa_migrate(p);
}

/*
 * Find out how many nodes the workload is actively running on. Do this by
 * tracking the nodes from which NUMA hinting faults are triggered. This can
 * be different from the set of nodes where the workload's memory is currently
 * located.
 */
/*
 * numa_group_count_active_nodes - 更新 NUMA 组最活跃节点的访问量及活跃节点数。
 */
static void numa_group_count_active_nodes(struct numa_group *numa_group)
{
	unsigned long faults, max_faults = 0;
	int nid, active_nodes = 0;

	for_each_node_state(nid, N_CPU) {
		faults = group_faults_cpu(numa_group, nid);
		if (faults > max_faults)
			max_faults = faults;
	}

	for_each_node_state(nid, N_CPU) {
		faults = group_faults_cpu(numa_group, nid);
		if (faults * ACTIVE_NODE_FRACTION > max_faults)
			active_nodes++;
	}

	numa_group->max_faults_cpu = max_faults;
	numa_group->active_nodes = active_nodes;
}

/*
 * When adapting the scan rate, the period is divided into NUMA_PERIOD_SLOTS
 * increments. The more local the fault statistics are, the higher the scan
 * period will be for the next scan window. If local/(local+remote) ratio is
 * below NUMA_PERIOD_THRESHOLD (where range of ratio is 1..NUMA_PERIOD_SLOTS)
 * the scan period will decrease. Aim for 70% local accesses.
 */
#define NUMA_PERIOD_SLOTS 10
#define NUMA_PERIOD_THRESHOLD 7

/*
 * Increase the scan period (slow down scanning) if the majority of
 * our memory is already on our local node, or if the majority of
 * the page accesses are shared with other processes.
 * Otherwise, decrease the scan period.
 */
/*
 * update_task_scan_period - 根据本地/远端、私有/共享访问比例调整扫描间隔。
 * 迁移失败或缺乏有效样本时退避，避免持续承担无收益的扫描成本。
 */
static void update_task_scan_period(struct task_struct *p,
			unsigned long shared, unsigned long private)
{
	unsigned int period_slot;
	int lr_ratio, ps_ratio;
	int diff;

	unsigned long remote = p->numa_faults_locality[0];
	unsigned long local = p->numa_faults_locality[1];

	/*
	 * If there were no record hinting faults then either the task is
	 * completely idle or all activity is in areas that are not of interest
	 * to automatic numa balancing. Related to that, if there were failed
	 * migration then it implies we are migrating too quickly or the local
	 * node is overloaded. In either case, scan slower
	 */
	if (local + shared == 0 || p->numa_faults_locality[2]) {
		p->numa_scan_period = min(p->numa_scan_period_max,
			p->numa_scan_period << 1);

		p->mm->numa_next_scan = jiffies +
			msecs_to_jiffies(p->numa_scan_period);

		return;
	}

	/*
	 * Prepare to scale scan period relative to the current period.
	 *	 == NUMA_PERIOD_THRESHOLD scan period stays the same
	 *       <  NUMA_PERIOD_THRESHOLD scan period decreases (scan faster)
	 *	 >= NUMA_PERIOD_THRESHOLD scan period increases (scan slower)
	 */
	period_slot = DIV_ROUND_UP(p->numa_scan_period, NUMA_PERIOD_SLOTS);
	lr_ratio = (local * NUMA_PERIOD_SLOTS) / (local + remote);
	ps_ratio = (private * NUMA_PERIOD_SLOTS) / (private + shared);

	if (ps_ratio >= NUMA_PERIOD_THRESHOLD) {
		/*
		 * Most memory accesses are local. There is no need to
		 * do fast NUMA scanning, since memory is already local.
		 */
		int slot = ps_ratio - NUMA_PERIOD_THRESHOLD;
		if (!slot)
			slot = 1;
		diff = slot * period_slot;
	} else if (lr_ratio >= NUMA_PERIOD_THRESHOLD) {
		/*
		 * Most memory accesses are shared with other tasks.
		 * There is no point in continuing fast NUMA scanning,
		 * since other tasks may just move the memory elsewhere.
		 */
		int slot = lr_ratio - NUMA_PERIOD_THRESHOLD;
		if (!slot)
			slot = 1;
		diff = slot * period_slot;
	} else {
		/*
		 * Private memory faults exceed (SLOTS-THRESHOLD)/SLOTS,
		 * yet they are not on the local NUMA node. Speed up
		 * NUMA scanning to get the memory moved over.
		 */
		int ratio = max(lr_ratio, ps_ratio);
		diff = -(NUMA_PERIOD_THRESHOLD - ratio) * period_slot;
	}

	p->numa_scan_period = clamp(p->numa_scan_period + diff,
			task_scan_min(p), task_scan_max(p));
	memset(p->numa_faults_locality, 0, sizeof(p->numa_faults_locality));
}

/*
 * Get the fraction of time the task has been running since the last
 * NUMA placement cycle. The scheduler keeps similar statistics, but
 * decays those on a 32ms period, which is orders of magnitude off
 * from the dozens-of-seconds NUMA balancing period. Use the scheduler
 * stats only if the task is so new there are no NUMA statistics yet.
 */
/*
 * numa_get_avg_runtime - 取得两次 NUMA 放置评估间的执行量和时间跨度，
 * 首次使用 PELT 估计。
 */
static u64 numa_get_avg_runtime(struct task_struct *p, u64 *period)
{
	u64 runtime, delta, now;
	/* Use the start of this time slice to avoid calculations. */
	now = p->se.exec_start;
	runtime = p->se.sum_exec_runtime;

	if (p->last_task_numa_placement) {
		delta = runtime - p->last_sum_exec_runtime;
		*period = now - p->last_task_numa_placement;

		/* Avoid time going backwards, prevent potential divide error: */
		if (unlikely((s64)*period < 0))
			*period = 0;
	} else {
		delta = p->se.avg.load_sum;
		*period = LOAD_AVG_MAX;
	}

	p->last_sum_exec_runtime = runtime;
	p->last_task_numa_placement = now;

	return delta;
}

/*
 * Determine the preferred nid for a task in a numa_group. This needs to
 * be done in a way that produces consistent results with group_weight,
 * otherwise workloads might not converge.
 */
/*
 * preferred_group_nid - 结合拓扑距离，
 * 为共享内存的 NUMA 组选择更合适的首选节点。
 */
static int preferred_group_nid(struct task_struct *p, int nid)
{
	nodemask_t nodes;
	int dist;

	/* Direct connections between all NUMA nodes. */
	if (sched_numa_topology_type == NUMA_DIRECT)
		return nid;

	/*
	 * On a system with glueless mesh NUMA topology, group_weight
	 * scores nodes according to the number of NUMA hinting faults on
	 * both the node itself, and on nearby nodes.
	 */
	if (sched_numa_topology_type == NUMA_GLUELESS_MESH) {
		unsigned long score, max_score = 0;
		int node, max_node = nid;

		dist = sched_max_numa_distance;

		for_each_node_state(node, N_CPU) {
			score = group_weight(p, node, dist);
			if (score > max_score) {
				max_score = score;
				max_node = node;
			}
		}
		return max_node;
	}

	/*
	 * Finding the preferred nid in a system with NUMA backplane
	 * interconnect topology is more involved. The goal is to locate
	 * tasks from numa_groups near each other in the system, and
	 * untangle workloads from different sides of the system. This requires
	 * searching down the hierarchy of node groups, recursively searching
	 * inside the highest scoring group of nodes. The nodemask tricks
	 * keep the complexity of the search down.
	 */
	nodes = node_states[N_CPU];
	for (dist = sched_max_numa_distance; dist > LOCAL_DISTANCE; dist--) {
		unsigned long max_faults = 0;
		nodemask_t max_group = NODE_MASK_NONE;
		int a, b;

		/* Are there nodes at this distance from each other? */
		if (!find_numa_distance(dist))
			continue;

		for_each_node_mask(a, nodes) {
			unsigned long faults = 0;
			nodemask_t this_group;
			nodes_clear(this_group);

			/* Sum group's NUMA faults; includes a==b case. */
			for_each_node_mask(b, nodes) {
				if (node_distance(a, b) < dist) {
					faults += group_faults(p, b);
					node_set(b, this_group);
					node_clear(b, nodes);
				}
			}

			/* Remember the top group. */
			if (faults > max_faults) {
				max_faults = faults;
				max_group = this_group;
				/*
				 * subtle: at the smallest distance there is
				 * just one node left in each "group", the
				 * winner is the preferred nid.
				 */
				nid = a;
			}
		}
		/* Next round, evaluate the nodes within max_group. */
		if (!max_faults)
			break;
		nodes = max_group;
	}
	return nid;
}

/*
 * task_numa_placement：
 * 每完成一轮扫描便衰减历史、合并新样本并更新首选节点和扫描周期。
 */
static void task_numa_placement(struct task_struct *p)
{
	int seq, nid, max_nid = NUMA_NO_NODE;
	unsigned long max_faults = 0;
	unsigned long fault_types[2] = { 0, 0 };
	unsigned long total_faults;
	u64 runtime, period;
	spinlock_t *group_lock = NULL;
	struct numa_group *ng;

	/*
	 * The p->mm->numa_scan_seq field gets updated without
	 * exclusive access. Use READ_ONCE() here to ensure
	 * that the field is read in a single access:
	 */
	seq = READ_ONCE(p->mm->numa_scan_seq);
	if (p->numa_scan_seq == seq)
		return;
	p->numa_scan_seq = seq;
	p->numa_scan_period_max = task_scan_max(p);

	total_faults = p->numa_faults_locality[0] +
		       p->numa_faults_locality[1];
	runtime = numa_get_avg_runtime(p, &period);

	/* If the task is part of a group prevent parallel updates to group stats */
	ng = deref_curr_numa_group(p);
	if (ng) {
		group_lock = &ng->lock;
		spin_lock_irq(group_lock);
	}

	/* Find the node with the highest number of faults */
	for_each_online_node(nid) {
		/* Keep track of the offsets in numa_faults array */
		int mem_idx, membuf_idx, cpu_idx, cpubuf_idx;
		unsigned long faults = 0, group_faults = 0;
		int priv;

		for (priv = 0; priv < NR_NUMA_HINT_FAULT_TYPES; priv++) {
			long diff, f_diff, f_weight;

			mem_idx = task_faults_idx(NUMA_MEM, nid, priv);
			membuf_idx = task_faults_idx(NUMA_MEMBUF, nid, priv);
			cpu_idx = task_faults_idx(NUMA_CPU, nid, priv);
			cpubuf_idx = task_faults_idx(NUMA_CPUBUF, nid, priv);

			/* Decay existing window, copy faults since last scan */
			diff = p->numa_faults[membuf_idx] - p->numa_faults[mem_idx] / 2;
			fault_types[priv] += p->numa_faults[membuf_idx];
			p->numa_faults[membuf_idx] = 0;

			/*
			 * Normalize the faults_from, so all tasks in a group
			 * count according to CPU use, instead of by the raw
			 * number of faults. Tasks with little runtime have
			 * little over-all impact on throughput, and thus their
			 * faults are less important.
			 */
			f_weight = div64_u64(runtime << 16, period + 1);
			f_weight = (f_weight * p->numa_faults[cpubuf_idx]) /
				   (total_faults + 1);
			f_diff = f_weight - p->numa_faults[cpu_idx] / 2;
			p->numa_faults[cpubuf_idx] = 0;

			p->numa_faults[mem_idx] += diff;
			p->numa_faults[cpu_idx] += f_diff;
			faults += p->numa_faults[mem_idx];
			p->total_numa_faults += diff;
			if (ng) {
				/*
				 * safe because we can only change our own group
				 *
				 * mem_idx represents the offset for a given
				 * nid and priv in a specific region because it
				 * is at the beginning of the numa_faults array.
				 */
				ng->faults[mem_idx] += diff;
				ng->faults[cpu_idx] += f_diff;
				ng->total_faults += diff;
				group_faults += ng->faults[mem_idx];
			}
		}

		if (!ng) {
			if (faults > max_faults) {
				max_faults = faults;
				max_nid = nid;
			}
		} else if (group_faults > max_faults) {
			max_faults = group_faults;
			max_nid = nid;
		}
	}

	/* Cannot migrate task to CPU-less node */
	max_nid = numa_nearest_node(max_nid, N_CPU);

	if (ng) {
		numa_group_count_active_nodes(ng);
		spin_unlock_irq(group_lock);
		max_nid = preferred_group_nid(p, max_nid);
	}

	if (max_faults) {
		/* Set the new preferred node */
		if (max_nid != p->numa_preferred_nid)
			sched_setnuma(p, max_nid);
	}

	update_task_scan_period(p, fault_types[0], fault_types[1]);
}

/*
 * get_numa_group - 仅在 NUMA 组尚未进入销毁流程时获取一个引用。
 */
static inline int get_numa_group(struct numa_group *grp)
{
	return refcount_inc_not_zero(&grp->refcount);
}

/*
 * put_numa_group - 释放 NUMA 组引用，最后一个引用消失后通过 RCU 延迟回收。
 */
static inline void put_numa_group(struct numa_group *grp)
{
	if (refcount_dec_and_test(&grp->refcount))
		kfree_rcu(grp, rcu);
}

/*
 * task_numa_group - 根据同页访问记录识别共享关系，
 * 并将任务并入合适的 NUMA 组。
 */
static void task_numa_group(struct task_struct *p, int cpupid, int flags,
			int *priv)
{
	struct numa_group *grp, *my_grp;
	struct task_struct *tsk;
	bool join = false;
	int cpu = cpupid_to_cpu(cpupid);
	int i;

	if (unlikely(!deref_curr_numa_group(p))) {
		unsigned int size = sizeof(struct numa_group) +
				    NR_NUMA_HINT_FAULT_STATS *
				    nr_node_ids * sizeof(unsigned long);

		grp = kzalloc(size, GFP_KERNEL | __GFP_NOWARN);
		if (!grp)
			return;

		refcount_set(&grp->refcount, 1);
		grp->active_nodes = 1;
		grp->max_faults_cpu = 0;
		spin_lock_init(&grp->lock);
		grp->gid = p->pid;

		for (i = 0; i < NR_NUMA_HINT_FAULT_STATS * nr_node_ids; i++)
			grp->faults[i] = p->numa_faults[i];

		grp->total_faults = p->total_numa_faults;

		grp->nr_tasks++;
		rcu_assign_pointer(p->numa_group, grp);
	}

	rcu_read_lock();
	tsk = READ_ONCE(cpu_rq(cpu)->curr);

	if (!cpupid_match_pid(tsk, cpupid))
		goto no_join;

	grp = rcu_dereference(tsk->numa_group);
	if (!grp)
		goto no_join;

	my_grp = deref_curr_numa_group(p);
	if (grp == my_grp)
		goto no_join;

	/*
	 * Only join the other group if its bigger; if we're the bigger group,
	 * the other task will join us.
	 */
	if (my_grp->nr_tasks > grp->nr_tasks)
		goto no_join;

	/*
	 * Tie-break on the grp address.
	 */
	if (my_grp->nr_tasks == grp->nr_tasks && my_grp > grp)
		goto no_join;

	/* Always join threads in the same process. */
	if (tsk->mm == current->mm)
		join = true;

	/* Simple filter to avoid false positives due to PID collisions */
	if (flags & TNF_SHARED)
		join = true;

	/* Update priv based on whether false sharing was detected */
	*priv = !join;

	if (join && !get_numa_group(grp))
		goto no_join;

	rcu_read_unlock();

	if (!join)
		return;

	WARN_ON_ONCE(irqs_disabled());
	double_lock_irq(&my_grp->lock, &grp->lock);

	for (i = 0; i < NR_NUMA_HINT_FAULT_STATS * nr_node_ids; i++) {
		my_grp->faults[i] -= p->numa_faults[i];
		grp->faults[i] += p->numa_faults[i];
	}
	my_grp->total_faults -= p->total_numa_faults;
	grp->total_faults += p->total_numa_faults;

	my_grp->nr_tasks--;
	grp->nr_tasks++;

	spin_unlock(&my_grp->lock);
	spin_unlock_irq(&grp->lock);

	rcu_assign_pointer(p->numa_group, grp);

	put_numa_group(my_grp);
	return;

no_join:
	rcu_read_unlock();
	return;
}

/*
 * Get rid of NUMA statistics associated with a task (either current or dead).
 * If @final is set, the task is dead and has reached refcount zero, so we can
 * safely free all relevant data structures. Otherwise, there might be
 * concurrent reads from places like load balancing and procfs, and we should
 * reset the data back to default state without freeing ->numa_faults.
 */
/*
 * task_numa_free - 撤销任务对 NUMA 组的统计贡献，
 * 按 final 决定释放还是重置统计。
 */
void task_numa_free(struct task_struct *p, bool final)
{
	/* safe: p either is current or is being freed by current */
	struct numa_group *grp = rcu_dereference_raw(p->numa_group);
	unsigned long *numa_faults = p->numa_faults;
	unsigned long flags;
	int i;

	if (!numa_faults)
		return;

	if (grp) {
		spin_lock_irqsave(&grp->lock, flags);
		for (i = 0; i < NR_NUMA_HINT_FAULT_STATS * nr_node_ids; i++)
			grp->faults[i] -= p->numa_faults[i];
		grp->total_faults -= p->total_numa_faults;

		grp->nr_tasks--;
		spin_unlock_irqrestore(&grp->lock, flags);
		RCU_INIT_POINTER(p->numa_group, NULL);
		put_numa_group(grp);
	}

	if (final) {
		p->numa_faults = NULL;
		kfree(numa_faults);
	} else {
		p->total_numa_faults = 0;
		for (i = 0; i < NR_NUMA_HINT_FAULT_STATS * nr_node_ids; i++)
			numa_faults[i] = 0;
	}
}

/*
 * Got a PROT_NONE fault for a page on @node.
 */
/*
 * task_numa_fault：
 * 记录一次 NUMA 提示故障反映的访问位置、共享关系和迁移结果。
 * 达到重试时刻时，更新放置决策并尝试任务迁移。
 */
void task_numa_fault(int last_cpupid, int mem_node, int pages, int flags)
{
	struct task_struct *p = current;
	bool migrated = flags & TNF_MIGRATED;
	int cpu_node = task_node(current);
	int local = !!(flags & TNF_FAULT_LOCAL);
	struct numa_group *ng;
	int priv;

	if (!static_branch_likely(&sched_numa_balancing))
		return;

	/* for example, ksmd faulting in a user's mm */
	if (!p->mm)
		return;

	/*
	 * NUMA faults statistics are unnecessary for the slow memory
	 * node for memory tiering mode.
	 */
	if (!node_is_toptier(mem_node) &&
	    (sysctl_numa_balancing_mode & NUMA_BALANCING_MEMORY_TIERING ||
	     !cpupid_valid(last_cpupid)))
		return;

	/* Allocate buffer to track faults on a per-node basis */
	if (unlikely(!p->numa_faults)) {
		int size = sizeof(*p->numa_faults) *
			   NR_NUMA_HINT_FAULT_BUCKETS * nr_node_ids;

		p->numa_faults = kzalloc(size, GFP_KERNEL|__GFP_NOWARN);
		if (!p->numa_faults)
			return;

		p->total_numa_faults = 0;
		memset(p->numa_faults_locality, 0, sizeof(p->numa_faults_locality));
	}

	/*
	 * First accesses are treated as private, otherwise consider accesses
	 * to be private if the accessing pid has not changed
	 */
	if (unlikely(last_cpupid == (-1 & LAST_CPUPID_MASK))) {
		priv = 1;
	} else {
		priv = cpupid_match_pid(p, last_cpupid);
		if (!priv && !(flags & TNF_NO_GROUP))
			task_numa_group(p, last_cpupid, flags, &priv);
	}

	/*
	 * If a workload spans multiple NUMA nodes, a shared fault that
	 * occurs wholly within the set of nodes that the workload is
	 * actively using should be counted as local. This allows the
	 * scan rate to slow down when a workload has settled down.
	 */
	ng = deref_curr_numa_group(p);
	if (!priv && !local && ng && ng->active_nodes > 1 &&
				numa_is_active_node(cpu_node, ng) &&
				numa_is_active_node(mem_node, ng))
		local = 1;

	/*
	 * Retry to migrate task to preferred node periodically, in case it
	 * previously failed, or the scheduler moved us.
	 */
	if (time_after(jiffies, p->numa_migrate_retry)) {
		task_numa_placement(p);
		numa_migrate_preferred(p);
	}

	if (migrated)
		p->numa_pages_migrated += pages;
	if (flags & TNF_MIGRATE_FAIL)
		p->numa_faults_locality[2] += pages;

	p->numa_faults[task_faults_idx(NUMA_MEMBUF, mem_node, priv)] += pages;
	p->numa_faults[task_faults_idx(NUMA_CPUBUF, cpu_node, priv)] += pages;
	p->numa_faults_locality[local] += pages;
}

/*
 * reset_ptenuma_scan - 递增共享地址空间的扫描轮次，
 * 并将扫描偏移重置到开头。
 */
static void reset_ptenuma_scan(struct task_struct *p)
{
	/*
	 * We only did a read acquisition of the mmap sem, so
	 * p->mm->numa_scan_seq is written to without exclusive access
	 * and the update is not guaranteed to be atomic. That's not
	 * much of an issue though, since this is just used for
	 * statistical sampling. Use READ_ONCE/WRITE_ONCE, which are not
	 * expensive, to avoid any form of compiler optimizations:
	 */
	WRITE_ONCE(p->mm->numa_scan_seq, READ_ONCE(p->mm->numa_scan_seq) + 1);
	p->mm->numa_scan_offset = 0;
}

/*
 * vma_is_accessed - 根据线程访问记录和扫描进度，
 * 判断当前线程是否应扫描此 VMA。
 */
static bool vma_is_accessed(struct mm_struct *mm, struct vm_area_struct *vma)
{
	unsigned long pids;
	/*
	 * Allow unconditional access first two times, so that all the (pages)
	 * of VMAs get prot_none fault introduced irrespective of accesses.
	 * This is also done to avoid any side effect of task scanning
	 * amplifying the unfairness of disjoint set of VMAs' access.
	 */
	if ((READ_ONCE(current->mm->numa_scan_seq) - vma->numab_state->start_scan_seq) < 2)
		return true;

	pids = vma->numab_state->pids_active[0] | vma->numab_state->pids_active[1];
	if (test_bit(hash_32(current->pid, ilog2(BITS_PER_LONG)), &pids))
		return true;

	/*
	 * Complete a scan that has already started regardless of PID access, or
	 * some VMAs may never be scanned in multi-threaded applications:
	 */
	if (mm->numa_scan_offset > vma->vm_start) {
		trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_IGNORE_PID);
		return true;
	}

	/*
	 * This vma has not been accessed for a while, and if the number
	 * the threads in the same process is low, which means no other
	 * threads can help scan this vma, force a vma scan.
	 */
	if (READ_ONCE(mm->numa_scan_seq) >
	   (vma->numab_state->prev_scan_seq + get_nr_threads(current)))
		return true;

	return false;
}

#define VMA_PID_RESET_PERIOD (4 * sysctl_numa_balancing_scan_delay)

/*
 * The expensive part of numa migration is done from task_work context.
 * Triggered from task_tick_numa().
 */
/*
 * task_numa_work - 在任务上下文分批扫描地址空间，
 * 设置 NUMA 提示故障所需的页表保护。
 * 利用 mm 级扫描时刻避免线程重复扫描，并限制每轮工作量。
 */
static void task_numa_work(struct callback_head *work)
{
	unsigned long migrate, next_scan, now = jiffies;
	struct task_struct *p = current;
	struct mm_struct *mm = p->mm;
	u64 runtime = p->se.sum_exec_runtime;
	struct vm_area_struct *vma;
	unsigned long start, end;
	unsigned long nr_pte_updates = 0;
	long pages, virtpages;
	struct vma_iterator vmi;
	bool vma_pids_skipped;
	bool vma_pids_forced = false;

	WARN_ON_ONCE(p != container_of(work, struct task_struct, numa_work));

	work->next = work;
	/*
	 * Who cares about NUMA placement when they're dying.
	 *
	 * NOTE: make sure not to dereference p->mm before this check,
	 * exit_task_work() happens _after_ exit_mm() so we could be called
	 * without p->mm even though we still had it when we enqueued this
	 * work.
	 */
	if (p->flags & PF_EXITING)
		return;

	/*
	 * Memory is pinned to only one NUMA node via cpuset.mems, naturally
	 * no page can be migrated.
	 */
	if (cpusets_enabled() && nodes_weight(cpuset_current_mems_allowed) == 1) {
		trace_sched_skip_cpuset_numa(current, &cpuset_current_mems_allowed);
		return;
	}

	if (!mm->numa_next_scan) {
		mm->numa_next_scan = now +
			msecs_to_jiffies(sysctl_numa_balancing_scan_delay);
	}

	/*
	 * Enforce maximal scan/migration frequency..
	 */
	migrate = mm->numa_next_scan;
	if (time_before(now, migrate))
		return;

	if (p->numa_scan_period == 0) {
		p->numa_scan_period_max = task_scan_max(p);
		p->numa_scan_period = task_scan_start(p);
	}

	next_scan = now + msecs_to_jiffies(p->numa_scan_period);
	/*
	 * 共享同一 mm 的线程竞争本轮扫描资格，只有更新成功者继续扫描。
	 */
	if (!try_cmpxchg(&mm->numa_next_scan, &migrate, next_scan))
		return;

	/*
	 * Delay this task enough that another task of this mm will likely win
	 * the next time around.
	 */
	p->node_stamp += 2 * TICK_NSEC;

	pages = sysctl_numa_balancing_scan_size;
	pages <<= 20 - PAGE_SHIFT; /* MB in pages */
	virtpages = pages * 8;	   /* Scan up to this much virtual space */
	if (!pages)
		return;


	if (!mmap_read_trylock(mm))
		return;

	/*
	 * VMAs are skipped if the current PID has not trapped a fault within
	 * the VMA recently. Allow scanning to be forced if there is no
	 * suitable VMA remaining.
	 */
	vma_pids_skipped = false;

retry_pids:
	start = mm->numa_scan_offset;
	vma_iter_init(&vmi, mm, start);
	vma = vma_next(&vmi);
	if (!vma) {
		reset_ptenuma_scan(p);
		start = 0;
		vma_iter_set(&vmi, start);
		vma = vma_next(&vmi);
	}

	for (; vma; vma = vma_next(&vmi)) {
		if (!vma_migratable(vma) || !vma_policy_mof(vma) ||
			is_vm_hugetlb_page(vma) || (vma->vm_flags & VM_MIXEDMAP)) {
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_UNSUITABLE);
			continue;
		}

		/*
		 * Shared library pages mapped by multiple processes are not
		 * migrated as it is expected they are cache replicated. Avoid
		 * hinting faults in read-only file-backed mappings or the vDSO
		 * as migrating the pages will be of marginal benefit.
		 */
		if (!vma->vm_mm ||
		    (vma->vm_file && (vma->vm_flags & (VM_READ|VM_WRITE)) == (VM_READ))) {
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_SHARED_RO);
			continue;
		}

		/*
		 * Skip inaccessible VMAs to avoid any confusion between
		 * PROT_NONE and NUMA hinting PTEs
		 */
		if (!vma_is_accessible(vma)) {
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_INACCESSIBLE);
			continue;
		}

		/* Initialise new per-VMA NUMAB state. */
		if (!vma->numab_state) {
			struct vma_numab_state *ptr;

			ptr = kzalloc(sizeof(*ptr), GFP_KERNEL);
			if (!ptr)
				continue;

			if (cmpxchg(&vma->numab_state, NULL, ptr)) {
				kfree(ptr);
				continue;
			}

			vma->numab_state->start_scan_seq = mm->numa_scan_seq;

			vma->numab_state->next_scan = now +
				msecs_to_jiffies(sysctl_numa_balancing_scan_delay);

			/* Reset happens after 4 times scan delay of scan start */
			vma->numab_state->pids_active_reset =  vma->numab_state->next_scan +
				msecs_to_jiffies(VMA_PID_RESET_PERIOD);

			/*
			 * Ensure prev_scan_seq does not match numa_scan_seq,
			 * to prevent VMAs being skipped prematurely on the
			 * first scan:
			 */
			 vma->numab_state->prev_scan_seq = mm->numa_scan_seq - 1;
		}

		/*
		 * Scanning the VMAs of short lived tasks add more overhead. So
		 * delay the scan for new VMAs.
		 */
		if (mm->numa_scan_seq && time_before(jiffies,
						vma->numab_state->next_scan)) {
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_SCAN_DELAY);
			continue;
		}

		/* RESET access PIDs regularly for old VMAs. */
		if (mm->numa_scan_seq &&
				time_after(jiffies, vma->numab_state->pids_active_reset)) {
			vma->numab_state->pids_active_reset = vma->numab_state->pids_active_reset +
				msecs_to_jiffies(VMA_PID_RESET_PERIOD);
			vma->numab_state->pids_active[0] = READ_ONCE(vma->numab_state->pids_active[1]);
			vma->numab_state->pids_active[1] = 0;
		}

		/* Do not rescan VMAs twice within the same sequence. */
		if (vma->numab_state->prev_scan_seq == mm->numa_scan_seq) {
			mm->numa_scan_offset = vma->vm_end;
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_SEQ_COMPLETED);
			continue;
		}

		/*
		 * Do not scan the VMA if task has not accessed it, unless no other
		 * VMA candidate exists.
		 */
		if (!vma_pids_forced && !vma_is_accessed(mm, vma)) {
			vma_pids_skipped = true;
			trace_sched_skip_vma_numa(mm, vma, NUMAB_SKIP_PID_INACTIVE);
			continue;
		}

		do {
			start = max(start, vma->vm_start);
			end = ALIGN(start + (pages << PAGE_SHIFT), HPAGE_SIZE);
			end = min(end, vma->vm_end);
			/*
			 * 本轮主要布置采样：
			 * 修改保护使后续访问产生提示故障，
			 * 随后再作迁移决策。
			 */
			nr_pte_updates = change_prot_numa(vma, start, end);

			/*
			 * Try to scan sysctl_numa_balancing_size worth of
			 * hpages that have at least one present PTE that
			 * is not already PTE-numa. If the VMA contains
			 * areas that are unused or already full of prot_numa
			 * PTEs, scan up to virtpages, to skip through those
			 * areas faster.
			 */
			if (nr_pte_updates)
				pages -= (end - start) >> PAGE_SHIFT;
			virtpages -= (end - start) >> PAGE_SHIFT;

			start = end;
			if (pages <= 0 || virtpages <= 0)
				goto out;

			cond_resched();
		} while (end != vma->vm_end);

		/* VMA scan is complete, do not scan until next sequence. */
		vma->numab_state->prev_scan_seq = mm->numa_scan_seq;

		/*
		 * Only force scan within one VMA at a time, to limit the
		 * cost of scanning a potentially uninteresting VMA.
		 */
		if (vma_pids_forced)
			break;
	}

	/*
	 * If no VMAs are remaining and VMAs were skipped due to the PID
	 * not accessing the VMA previously, then force a scan to ensure
	 * forward progress:
	 */
	if (!vma && !vma_pids_forced && vma_pids_skipped) {
		vma_pids_forced = true;
		goto retry_pids;
	}

out:
	/*
	 * It is possible to reach the end of the VMA list but the last few
	 * VMAs are not guaranteed to the vma_migratable. If they are not, we
	 * would find the !migratable VMA on the next scan but not reset the
	 * scanner to the start so check it now.
	 */
	if (vma)
		mm->numa_scan_offset = start;
	else
		reset_ptenuma_scan(p);
	mmap_read_unlock(mm);

	/*
	 * Make sure tasks use at least 32x as much time to run other code
	 * than they used here, to limit NUMA PTE scanning overhead to 3% max.
	 * Usually update_task_scan_period slows down scanning enough; on an
	 * overloaded system we need to limit overhead on a per task basis.
	 */
	if (unlikely(p->se.sum_exec_runtime != runtime)) {
		u64 diff = p->se.sum_exec_runtime - runtime;
		p->node_stamp += 32 * diff;
	}
}

/*
 * init_numa_balancing - 初始化新任务的 NUMA 统计和扫描工作，
 * 错开共享地址空间的扫描时机。
 */
void init_numa_balancing(u64 clone_flags, struct task_struct *p)
{
	int mm_users = 0;
	struct mm_struct *mm = p->mm;

	if (mm) {
		mm_users = atomic_read(&mm->mm_users);
		if (mm_users == 1) {
			mm->numa_next_scan = jiffies + msecs_to_jiffies(sysctl_numa_balancing_scan_delay);
			mm->numa_scan_seq = 0;
		}
	}
	p->node_stamp			= 0;
	p->numa_scan_seq		= mm ? mm->numa_scan_seq : 0;
	p->numa_scan_period		= sysctl_numa_balancing_scan_delay;
	p->numa_migrate_retry		= 0;
	/* Protect against double add, see task_tick_numa and task_numa_work */
	p->numa_work.next		= &p->numa_work;
	p->numa_faults			= NULL;
	p->numa_pages_migrated		= 0;
	p->total_numa_faults		= 0;
	RCU_INIT_POINTER(p->numa_group, NULL);
	p->last_task_numa_placement	= 0;
	p->last_sum_exec_runtime	= 0;

	init_task_work(&p->numa_work, task_numa_work);

	/* New address space, reset the preferred nid */
	if (!(clone_flags & CLONE_VM)) {
		p->numa_preferred_nid = NUMA_NO_NODE;
		return;
	}

	/*
	 * New thread, keep existing numa_preferred_nid which should be copied
	 * already by arch_dup_task_struct but stagger when scans start.
	 */
	if (mm) {
		unsigned int delay;

		delay = min_t(unsigned int, task_scan_max(current),
			current->numa_scan_period * mm_users * NSEC_PER_MSEC);
		delay += 2 * TICK_NSEC;
		p->node_stamp = delay;
	}
}

/*
 * Drive the periodic memory faults..
 */
/*
 * task_tick_numa - 周期检查执行量，
 * 将到期的 NUMA 扫描排入返回用户态时执行的 task_work。
 */
static void task_tick_numa(struct rq *rq, struct task_struct *curr)
{
	struct callback_head *work = &curr->numa_work;
	u64 period, now;

	/*
	 * We don't care about NUMA placement if we don't have memory.
	 */
	if (!curr->mm || (curr->flags & (PF_EXITING | PF_KTHREAD)) || work->next != work)
		return;

	/*
	 * Using runtime rather than walltime has the dual advantage that
	 * we (mostly) drive the selection from busy threads and that the
	 * task needs to have done some actual work before we bother with
	 * NUMA placement.
	 */
	now = curr->se.sum_exec_runtime;
	period = (u64)curr->numa_scan_period * NSEC_PER_MSEC;

	if (now > curr->node_stamp + period) {
		if (!curr->node_stamp)
			curr->numa_scan_period = task_scan_start(curr);
		curr->node_stamp += period;

		if (!time_before(jiffies, curr->mm->numa_next_scan))
			task_work_add(curr, work, TWA_RESUME);
	}
}

/*
 * update_scan_period - 跨节点迁移破坏既有放置关系时，
 * 重设扫描间隔以重新学习局部性。
 */
static void update_scan_period(struct task_struct *p, int new_cpu)
{
	int src_nid = cpu_to_node(task_cpu(p));
	int dst_nid = cpu_to_node(new_cpu);

	if (!static_branch_likely(&sched_numa_balancing))
		return;

	if (!p->mm || !p->numa_faults || (p->flags & PF_EXITING))
		return;

	if (src_nid == dst_nid)
		return;

	/*
	 * Allow resets if faults have been trapped before one scan
	 * has completed. This is most likely due to a new task that
	 * is pulled cross-node due to wakeups or load balancing.
	 */
	if (p->numa_scan_seq) {
		/*
		 * Avoid scan adjustments if moving to the preferred
		 * node or if the task was not previously running on
		 * the preferred node.
		 */
		if (dst_nid == p->numa_preferred_nid ||
		    (p->numa_preferred_nid != NUMA_NO_NODE &&
			src_nid != p->numa_preferred_nid))
			return;
	}

	p->numa_scan_period = task_scan_start(p);
}

#else /* !CONFIG_NUMA_BALANCING: */

/*
 * task_tick_numa - 未启用 NUMA 自动均衡时，不安排扫描工作。
 */
static void task_tick_numa(struct rq *rq, struct task_struct *curr)
{
}

/*
 * account_numa_enqueue - 未启用 NUMA 自动均衡时，无需更新入队统计。
 */
static inline void account_numa_enqueue(struct rq *rq, struct task_struct *p)
{
}

/*
 * account_numa_dequeue - 未启用 NUMA 自动均衡时，无需更新出队统计。
 */
static inline void account_numa_dequeue(struct rq *rq, struct task_struct *p)
{
}

/*
 * update_scan_period - 未启用 NUMA 自动均衡时，无需调整扫描周期。
 */
static inline void update_scan_period(struct task_struct *p, int new_cpu)
{
}

#endif /* !CONFIG_NUMA_BALANCING */

/*
 * 实体队列统计与 PELT 历史统计分别维护：实体离队后，
 * 历史贡献仍可能继续衰减。
 * 后续 load_avg 按权重计负载，runnable_avg 计可运行需求，
 * util_avg 计实际执行。
 */
/*
 * account_entity_enqueue - 累计实体权重和队列数量，
 * 并为任务维护迁移链表及 NUMA 计数。
 */
static void
account_entity_enqueue(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	update_load_add(&cfs_rq->load, se->load.weight);
	if (entity_is_task(se)) {
		struct rq *rq = rq_of(cfs_rq);

		account_numa_enqueue(rq, task_of(se));
		list_add(&se->group_node, &rq->cfs_tasks);
	}
	cfs_rq->nr_queued++;
}

/*
 * account_entity_dequeue - 撤销实体的权重、队列计数及任务链表记录。
 */
static void
account_entity_dequeue(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	update_load_sub(&cfs_rq->load, se->load.weight);
	if (entity_is_task(se)) {
		account_numa_dequeue(rq_of(cfs_rq), task_of(se));
		list_del_init(&se->group_node);
	}
	cfs_rq->nr_queued--;
}

/*
 * Signed add and clamp on underflow.
 *
 * Explicitly do a load-store to ensure the intermediate value never hits
 * memory. This allows lockless observations without ever seeing the negative
 * values.
 */
#define add_positive(_ptr, _val) do {                           \
	typeof(_ptr) ptr = (_ptr);                              \
	typeof(_val) val = (_val);                              \
	typeof(*ptr) res, var = READ_ONCE(*ptr);                \
								\
	res = var + val;                                        \
								\
	if (val < 0 && res > var)                               \
		res = 0;                                        \
								\
	WRITE_ONCE(*ptr, res);                                  \
} while (0)

/*
 * Unsigned subtract and clamp on underflow.
 *
 * Explicitly do a load-store to ensure the intermediate value never hits
 * memory. This allows lockless observations without ever seeing the negative
 * values.
 */
#define sub_positive(_ptr, _val) do {				\
	typeof(_ptr) ptr = (_ptr);				\
	typeof(*ptr) val = (_val);				\
	typeof(*ptr) res, var = READ_ONCE(*ptr);		\
	res = var - val;					\
	if (res > var)						\
		res = 0;					\
	WRITE_ONCE(*ptr, res);					\
} while (0)

/*
 * Remove and clamp on negative, from a local variable.
 *
 * A variant of sub_positive(), which does not use explicit load-store
 * and is thus optimized for local variable updates.
 */
#define lsub_positive(_ptr, _val) do {				\
	typeof(_ptr) ptr = (_ptr);				\
	*ptr -= min_t(typeof(*ptr), *ptr, _val);		\
} while (0)

/*
 * enqueue_load_avg - 把实体的 PELT 负载均值与加权累计量加入队列。
 */
static inline void
enqueue_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	cfs_rq->avg.load_avg += se->avg.load_avg;
	cfs_rq->avg.load_sum += se_weight(se) * se->avg.load_sum;
}

/*
 * dequeue_load_avg - 从队列减去实体负载，并限制舍入误差导致的累计量下溢。
 */
static inline void
dequeue_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	sub_positive(&cfs_rq->avg.load_avg, se->avg.load_avg);
	sub_positive(&cfs_rq->avg.load_sum, se_weight(se) * se->avg.load_sum);
	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.load_sum = max_t(u32, cfs_rq->avg.load_sum,
					  cfs_rq->avg.load_avg * PELT_MIN_DIVIDER);
}

/*
 * rescale_entity - 按新旧权重之比缩放虚拟 lag、相对截止时间及保护边界。
 * 换权重时保留服务欠账，而不是直接丢弃原有公平性状态。
 */
static void
rescale_entity(struct sched_entity *se, unsigned long weight, bool rel_vprot)
{
	unsigned long old_weight = se->load.weight;

	/*
	 * VRUNTIME
	 * --------
	 *
	 * COROLLARY #1: The virtual runtime of the entity needs to be
	 * adjusted if re-weight at !0-lag point.
	 *
	 * Proof: For contradiction assume this is not true, so we can
	 * re-weight without changing vruntime at !0-lag point.
	 *
	 *             Weight	VRuntime   Avg-VRuntime
	 *     before    w          v            V
	 *      after    w'         v'           V'
	 *
	 * Since lag needs to be preserved through re-weight:
	 *
	 *	lag = (V - v)*w = (V'- v')*w', where v = v'
	 *	==>	V' = (V - v)*w/w' + v		(1)
	 *
	 * Let W be the total weight of the entities before reweight,
	 * since V' is the new weighted average of entities:
	 *
	 *	V' = (WV + w'v - wv) / (W + w' - w)	(2)
	 *
	 * by using (1) & (2) we obtain:
	 *
	 *	(WV + w'v - wv) / (W + w' - w) = (V - v)*w/w' + v
	 *	==> (WV-Wv+Wv+w'v-wv)/(W+w'-w) = (V - v)*w/w' + v
	 *	==> (WV - Wv)/(W + w' - w) + v = (V - v)*w/w' + v
	 *	==>	(V - v)*W/(W + w' - w) = (V - v)*w/w' (3)
	 *
	 * Since we are doing at !0-lag point which means V != v, we
	 * can simplify (3):
	 *
	 *	==>	W / (W + w' - w) = w / w'
	 *	==>	Ww' = Ww + ww' - ww
	 *	==>	W * (w' - w) = w * (w' - w)
	 *	==>	W = w	(re-weight indicates w' != w)
	 *
	 * So the cfs_rq contains only one entity, hence vruntime of
	 * the entity @v should always equal to the cfs_rq's weighted
	 * average vruntime @V, which means we will always re-weight
	 * at 0-lag point, thus breach assumption. Proof completed.
	 *
	 *
	 * COROLLARY #2: Re-weight does NOT affect weighted average
	 * vruntime of all the entities.
	 *
	 * Proof: According to corollary #1, Eq. (1) should be:
	 *
	 *	(V - v)*w = (V' - v')*w'
	 *	==>    v' = V' - (V - v)*w/w'		(4)
	 *
	 * According to the weighted average formula, we have:
	 *
	 *	V' = (WV - wv + w'v') / (W - w + w')
	 *	   = (WV - wv + w'(V' - (V - v)w/w')) / (W - w + w')
	 *	   = (WV - wv + w'V' - Vw + wv) / (W - w + w')
	 *	   = (WV + w'V' - Vw) / (W - w + w')
	 *
	 *	==>  V'*(W - w + w') = WV + w'V' - Vw
	 *	==>	V' * (W - w) = (W - w) * V	(5)
	 *
	 * If the entity is the only one in the cfs_rq, then reweight
	 * always occurs at 0-lag point, so V won't change. Or else
	 * there are other entities, hence W != w, then Eq. (5) turns
	 * into V' = V. So V won't change in either case, proof done.
	 *
	 *
	 * So according to corollary #1 & #2, the effect of re-weight
	 * on vruntime should be:
	 *
	 *	v' = V' - (V - v) * w / w'		(4)
	 *	   = V  - (V - v) * w / w'
	 *	   = V  - vl * w / w'
	 *	   = V  - vl'
	 */
	/*
	 * 按旧权重/新权重缩放虚拟欠账，
	 * 使其对应的加权服务欠账保持连续。
	 */
	se->vlag = div64_long(se->vlag * old_weight, weight);

	/*
	 * DEADLINE
	 * --------
	 *
	 * When the weight changes, the virtual time slope changes and
	 * we should adjust the relative virtual deadline accordingly.
	 *
	 *	d' = v' + (d - v)*w/w'
	 *	   = V' - (V - v)*w/w' + (d - v)*w/w'
	 *	   = V  - (V - v)*w/w' + (d - v)*w/w'
	 *	   = V  + (d - V)*w/w'
	 */
	if (se->rel_deadline)
		se->deadline = div64_long(se->deadline * old_weight, weight);

	if (rel_vprot)
		se->vprot = div64_long(se->vprot * old_weight, weight);
}

/*
 * reweight_entity - 更新实体权重，同时重算虚拟时间和 PELT 负载。
 * 在队实体先撤销旧贡献，调整后再加入，以保持队列统计一致。
 */
static void reweight_entity(struct cfs_rq *cfs_rq, struct sched_entity *se,
			    unsigned long weight)
{
	bool curr = cfs_rq->curr == se;
	bool rel_vprot = false;
	u64 avruntime = 0;

	if (se->on_rq) {
		/* commit outstanding execution time */
		update_curr(cfs_rq);
		avruntime = avg_vruntime(cfs_rq);
		se->vlag = entity_lag(cfs_rq, se, avruntime);
		se->deadline -= avruntime;
		se->rel_deadline = 1;
		if (curr && protect_slice(se)) {
			se->vprot -= avruntime;
			rel_vprot = true;
		}

		cfs_rq->nr_queued--;
		if (!curr)
			__dequeue_entity(cfs_rq, se);
		update_load_sub(&cfs_rq->load, se->load.weight);
	}
	dequeue_load_avg(cfs_rq, se);

	rescale_entity(se, weight, rel_vprot);

	update_load_set(&se->load, weight);

	do {
		u32 divider = get_pelt_divider(&se->avg);
		se->avg.load_avg = div_u64(se_weight(se) * se->avg.load_sum, divider);
	} while (0);

	enqueue_load_avg(cfs_rq, se);
	if (se->on_rq) {
		if (rel_vprot)
			se->vprot += avruntime;
		se->deadline += avruntime;
		se->rel_deadline = 0;
		se->vruntime = avruntime - se->vlag;

		update_load_add(&cfs_rq->load, se->load.weight);
		if (!curr)
			__enqueue_entity(cfs_rq, se);
		cfs_rq->nr_queued++;
	}
}

/*
 * reweight_task_fair - 应用任务的新调度权重，并更新对应的倒数缓存。
 */
static void reweight_task_fair(struct rq *rq, struct task_struct *p,
			       const struct load_weight *lw)
{
	struct sched_entity *se = &p->se;
	struct cfs_rq *cfs_rq = cfs_rq_of(se);
	struct load_weight *load = &se->load;

	reweight_entity(cfs_rq, se, lw->weight);
	load->inv_weight = lw->inv_weight;
}

static inline int throttled_hierarchy(struct cfs_rq *cfs_rq);

#ifdef CONFIG_FAIR_GROUP_SCHED
/*
 * All this does is approximate the hierarchical proportion which includes that
 * global sum we all love to hate.
 *
 * That is, the weight of a group entity, is the proportional share of the
 * group weight based on the group runqueue weights. That is:
 *
 *                     tg->weight * grq->load.weight
 *   ge->load.weight = -----------------------------               (1)
 *                       \Sum grq->load.weight
 *
 * Now, because computing that sum is prohibitively expensive to compute (been
 * there, done that) we approximate it with this average stuff. The average
 * moves slower and therefore the approximation is cheaper and more stable.
 *
 * So instead of the above, we substitute:
 *
 *   grq->load.weight -> grq->avg.load_avg                         (2)
 *
 * which yields the following:
 *
 *                     tg->weight * grq->avg.load_avg
 *   ge->load.weight = ------------------------------              (3)
 *                             tg->load_avg
 *
 * Where: tg->load_avg ~= \Sum grq->avg.load_avg
 *
 * That is shares_avg, and it is right (given the approximation (2)).
 *
 * The problem with it is that because the average is slow -- it was designed
 * to be exactly that of course -- this leads to transients in boundary
 * conditions. In specific, the case where the group was idle and we start the
 * one task. It takes time for our CPU's grq->avg.load_avg to build up,
 * yielding bad latency etc..
 *
 * Now, in that special case (1) reduces to:
 *
 *                     tg->weight * grq->load.weight
 *   ge->load.weight = ----------------------------- = tg->weight   (4)
 *                         grp->load.weight
 *
 * That is, the sum collapses because all other CPUs are idle; the UP scenario.
 *
 * So what we do is modify our approximation (3) to approach (4) in the (near)
 * UP case, like:
 *
 *   ge->load.weight =
 *
 *              tg->weight * grq->load.weight
 *     ---------------------------------------------------         (5)
 *     tg->load_avg - grq->avg.load_avg + grq->load.weight
 *
 * But because grq->load.weight can drop to 0, resulting in a divide by zero,
 * we need to use grq->avg.load_avg as its lower bound, which then gives:
 *
 *
 *                     tg->weight * grq->load.weight
 *   ge->load.weight = -----------------------------		   (6)
 *                             tg_load_avg'
 *
 * Where:
 *
 *   tg_load_avg' = tg->load_avg - grq->avg.load_avg +
 *                  max(grq->load.weight, grq->avg.load_avg)
 *
 * And that is shares_weight and is icky. In the (near) UP case it approaches
 * (4) while in the normal case it approaches (3). It consistently
 * overestimates the ge->load.weight and therefore:
 *
 *   \Sum ge->load.weight >= tg->weight
 *
 * hence icky!
 */
/*
 * calc_group_shares - 按本 CPU 在任务组总负载中的占比，
 * 估算组实体应获得的权重。
 */
static long calc_group_shares(struct cfs_rq *cfs_rq)
{
	long tg_weight, tg_shares, load, shares;
	struct task_group *tg = cfs_rq->tg;

	tg_shares = READ_ONCE(tg->shares);

	/*
	 * 瞬时负载能及时反映刚唤醒的任务，PELT 均值则保留历史需求，
	 * 二者取大。
	 */
	load = max(scale_load_down(cfs_rq->load.weight), cfs_rq->avg.load_avg);

	tg_weight = atomic_long_read(&tg->load_avg);

	/* Ensure tg_weight >= load */
	tg_weight -= cfs_rq->tg_load_avg_contrib;
	tg_weight += load;

	shares = (tg_shares * load);
	if (tg_weight)
		shares /= tg_weight;

	/*
	 * MIN_SHARES has to be unscaled here to support per-CPU partitioning
	 * of a group with small tg->shares value. It is a floor value which is
	 * assigned as a minimum load.weight to the sched_entity representing
	 * the group on a CPU.
	 *
	 * E.g. on 64-bit for a group with tg->shares of scale_load(15)=15*1024
	 * on an 8-core system with 8 tasks each runnable on one CPU shares has
	 * to be 15*1024*1/8=1920 instead of scale_load(MIN_SHARES)=2*1024. In
	 * case no task is runnable on a CPU MIN_SHARES=2 should be returned
	 * instead of 0.
	 */
	return clamp_t(long, shares, MIN_SHARES, tg_shares);
}

/*
 * Recomputes the group entity based on the current state of its group
 * runqueue.
 */
/*
 * update_cfs_group - 当组内有负载时重算其在父队列中的权重。
 */
static void update_cfs_group(struct sched_entity *se)
{
	struct cfs_rq *gcfs_rq = group_cfs_rq(se);
	long shares;

	/*
	 * When a group becomes empty, preserve its weight. This matters for
	 * DELAY_DEQUEUE.
	 */
	if (!gcfs_rq || !gcfs_rq->load.weight)
		return;

	shares = calc_group_shares(gcfs_rq);
	if (unlikely(se->load.weight != shares))
		reweight_entity(cfs_rq_of(se), se, shares);
}

#else /* !CONFIG_FAIR_GROUP_SCHED: */
/*
 * update_cfs_group - 未启用组调度时，无需把组份额折算为实体权重。
 */
static inline void update_cfs_group(struct sched_entity *se)
{
}
#endif /* !CONFIG_FAIR_GROUP_SCHED */

/*
 * cfs_rq_util_change - 仅在根公平队列利用率变化时通知 CPU 调频路径。
 */
static inline void cfs_rq_util_change(struct cfs_rq *cfs_rq, int flags)
{
	struct rq *rq = rq_of(cfs_rq);

	if (&rq->cfs == cfs_rq) {
		/*
		 * There are a few boundary cases this might miss but it should
		 * get called often enough that that should (hopefully) not be
		 * a real problem.
		 *
		 * It will not get called when we go idle, because the idle
		 * thread is a different class (!fair), nor will the utilization
		 * number include things like RT tasks.
		 *
		 * As is, the util number is not freq-invariant (we'd have to
		 * implement arch_scale_freq_capacity() for that).
		 *
		 * See cpu_util_cfs().
		 */
		cpufreq_update_util(rq, flags);
	}
}

/*
 * load_avg_is_decayed - 判断 PELT 三类累计量是否都已衰减为零，
 * 并检查均值一致性。
 */
static inline bool load_avg_is_decayed(struct sched_avg *sa)
{
	if (sa->load_sum)
		return false;

	if (sa->util_sum)
		return false;

	if (sa->runnable_sum)
		return false;

	/*
	 * _avg must be null when _sum are null because _avg = _sum / divider
	 * Make sure that rounding and/or propagation of PELT values never
	 * break this.
	 */
	WARN_ON_ONCE(sa->load_avg ||
		      sa->util_avg ||
		      sa->runnable_avg);

	return true;
}

/*
 * cfs_rq_last_update_time - 读取队列 PELT 时间戳，兼顾 32 位平台上的一致性。
 */
static inline u64 cfs_rq_last_update_time(struct cfs_rq *cfs_rq)
{
	return u64_u32_load_copy(cfs_rq->avg.last_update_time,
				 cfs_rq->last_update_time_copy);
}
#ifdef CONFIG_FAIR_GROUP_SCHED
/*
 * Because list_add_leaf_cfs_rq always places a child cfs_rq on the list
 * immediately before a parent cfs_rq, and cfs_rqs are removed from the list
 * bottom-up, we only have to test whether the cfs_rq before us on the list
 * is our child.
 * If cfs_rq is not on the list, test whether a child needs its to be added to
 * connect a branch to the tree  * (see list_add_leaf_cfs_rq() for details).
 */
/*
 * child_cfs_rq_on_list - 利用子先父后的链表顺序，
 * 判断队列是否仍有子队列需要维护。
 */
static inline bool child_cfs_rq_on_list(struct cfs_rq *cfs_rq)
{
	struct cfs_rq *prev_cfs_rq;
	struct list_head *prev;
	struct rq *rq = rq_of(cfs_rq);

	if (cfs_rq->on_list) {
		prev = cfs_rq->leaf_cfs_rq_list.prev;
	} else {
		prev = rq->tmp_alone_branch;
	}

	if (prev == &rq->leaf_cfs_rq_list)
		return false;

	prev_cfs_rq = container_of(prev, struct cfs_rq, leaf_cfs_rq_list);

	return (prev_cfs_rq->tg->parent == cfs_rq->tg);
}

/*
 * cfs_rq_is_decayed - 判断队列自身、子层级及组贡献是否均已清空，
 * 可停止周期维护。
 */
static inline bool cfs_rq_is_decayed(struct cfs_rq *cfs_rq)
{
	if (cfs_rq->load.weight)
		return false;

	if (!load_avg_is_decayed(&cfs_rq->avg))
		return false;

	if (child_cfs_rq_on_list(cfs_rq))
		return false;

	if (cfs_rq->tg_load_avg_contrib)
		return false;

	return true;
}

/*
 * update_tg_load_avg - 把本 CPU 的负载变化增量汇入任务组全局值。
 * 设置最短更新间隔和变化阈值，减少跨 CPU 共享缓存行争用。
 */
/**
 * update_tg_load_avg - update the tg's load avg
 * @cfs_rq: the cfs_rq whose avg changed
 *
 * This function 'ensures': tg->load_avg := \Sum tg->cfs_rq[]->avg.load.
 * However, because tg->load_avg is a global value there are performance
 * considerations.
 *
 * In order to avoid having to look at the other cfs_rq's, we use a
 * differential update where we store the last value we propagated. This in
 * turn allows skipping updates if the differential is 'small'.
 *
 * Updating tg's load_avg is necessary before update_cfs_share().
 */
static inline void update_tg_load_avg(struct cfs_rq *cfs_rq)
{
	long delta;
	u64 now;

	/*
	 * No need to update load_avg for root_task_group as it is not used.
	 */
	if (cfs_rq->tg == &root_task_group)
		return;

	/* rq has been offline and doesn't contribute to the share anymore: */
	if (!cpu_active(cpu_of(rq_of(cfs_rq))))
		return;

	/*
	 * For migration heavy workloads, access to tg->load_avg can be
	 * unbound. Limit the update rate to at most once per ms.
	 */
	now = sched_clock_cpu(cpu_of(rq_of(cfs_rq)));
	if (now - cfs_rq->last_update_tg_load_avg < NSEC_PER_MSEC)
		return;

	delta = cfs_rq->avg.load_avg - cfs_rq->tg_load_avg_contrib;
	if (abs(delta) > cfs_rq->tg_load_avg_contrib / 64) {
		atomic_long_add(delta, &cfs_rq->tg->load_avg);
		cfs_rq->tg_load_avg_contrib = cfs_rq->avg.load_avg;
		cfs_rq->last_update_tg_load_avg = now;
	}
}

/*
 * clear_tg_load_avg - 从任务组全局负载中撤销本队列上次报告的贡献。
 */
static inline void clear_tg_load_avg(struct cfs_rq *cfs_rq)
{
	long delta;
	u64 now;

	/*
	 * No need to update load_avg for root_task_group, as it is not used.
	 */
	if (cfs_rq->tg == &root_task_group)
		return;

	now = sched_clock_cpu(cpu_of(rq_of(cfs_rq)));
	delta = 0 - cfs_rq->tg_load_avg_contrib;
	atomic_long_add(delta, &cfs_rq->tg->load_avg);
	cfs_rq->tg_load_avg_contrib = 0;
	cfs_rq->last_update_tg_load_avg = now;
}

/* CPU offline callback: */
/*
 * clear_tg_offline_cfs_rqs - CPU 离线时清除其各任务组队列对全局负载的贡献。
 */
static void __maybe_unused clear_tg_offline_cfs_rqs(struct rq *rq)
{
	struct task_group *tg;

	lockdep_assert_rq_held(rq);

	/*
	 * The rq clock has already been updated in
	 * set_rq_offline(), so we should skip updating
	 * the rq clock again in unthrottle_cfs_rq().
	 */
	rq_clock_start_loop_update(rq);

	rcu_read_lock();
	list_for_each_entry_rcu(tg, &task_groups, list) {
		struct cfs_rq *cfs_rq = tg->cfs_rq[cpu_of(rq)];

		clear_tg_load_avg(cfs_rq);
	}
	rcu_read_unlock();

	rq_clock_stop_loop_update(rq);
}

/*
 * Called within set_task_rq() right before setting a task's CPU. The
 * caller only guarantees p->pi_lock is held; no other assumptions,
 * including the state of rq->lock, should be made.
 */
/*
 * set_task_rq_fair - 切换所属队列前衰减实体负载，
 * 再对齐到目标队列的 PELT 时间基准。
 */
void set_task_rq_fair(struct sched_entity *se,
		      struct cfs_rq *prev, struct cfs_rq *next)
{
	u64 p_last_update_time;
	u64 n_last_update_time;

	if (!sched_feat(ATTACH_AGE_LOAD))
		return;

	/*
	 * We are supposed to update the task to "current" time, then its up to
	 * date and ready to go to new CPU/cfs_rq. But we have difficulty in
	 * getting what current time is, so simply throw away the out-of-date
	 * time. This will result in the wakee task is less decayed, but giving
	 * the wakee more load sounds not bad.
	 */
	if (!(se->avg.last_update_time && prev))
		return;

	p_last_update_time = cfs_rq_last_update_time(prev);
	n_last_update_time = cfs_rq_last_update_time(next);

	__update_load_avg_blocked_se(p_last_update_time, se);
	se->avg.last_update_time = n_last_update_time;
}

/*
 * When on migration a sched_entity joins/leaves the PELT hierarchy, we need to
 * propagate its contribution. The key to this propagation is the invariant
 * that for each group:
 *
 *   ge->avg == grq->avg						(1)
 *
 * _IFF_ we look at the pure running and runnable sums. Because they
 * represent the very same entity, just at different points in the hierarchy.
 *
 * Per the above update_tg_cfs_util() and update_tg_cfs_runnable() are trivial
 * and simply copies the running/runnable sum over (but still wrong, because
 * the group entity and group rq do not have their PELT windows aligned).
 *
 * However, update_tg_cfs_load() is more complex. So we have:
 *
 *   ge->avg.load_avg = ge->load.weight * ge->avg.runnable_avg		(2)
 *
 * And since, like util, the runnable part should be directly transferable,
 * the following would _appear_ to be the straight forward approach:
 *
 *   grq->avg.load_avg = grq->load.weight * grq->avg.runnable_avg	(3)
 *
 * And per (1) we have:
 *
 *   ge->avg.runnable_avg == grq->avg.runnable_avg
 *
 * Which gives:
 *
 *                      ge->load.weight * grq->avg.load_avg
 *   ge->avg.load_avg = -----------------------------------		(4)
 *                               grq->load.weight
 *
 * Except that is wrong!
 *
 * Because while for entities historical weight is not important and we
 * really only care about our future and therefore can consider a pure
 * runnable sum, runqueues can NOT do this.
 *
 * We specifically want runqueues to have a load_avg that includes
 * historical weights. Those represent the blocked load, the load we expect
 * to (shortly) return to us. This only works by keeping the weights as
 * integral part of the sum. We therefore cannot decompose as per (3).
 *
 * Another reason this doesn't work is that runnable isn't a 0-sum entity.
 * Imagine a rq with 2 tasks that each are runnable 2/3 of the time. Then the
 * rq itself is runnable anywhere between 2/3 and 1 depending on how the
 * runnable section of these tasks overlap (or not). If they were to perfectly
 * align the rq as a whole would be runnable 2/3 of the time. If however we
 * always have at least 1 runnable task, the rq as a whole is always runnable.
 *
 * So we'll have to approximate.. :/
 *
 * Given the constraint:
 *
 *   ge->avg.running_sum <= ge->avg.runnable_sum <= LOAD_AVG_MAX
 *
 * We can construct a rule that adds runnable to a rq by assuming minimal
 * overlap.
 *
 * On removal, we'll assume each task is equally runnable; which yields:
 *
 *   grq->avg.runnable_sum = grq->avg.load_sum / grq->load.weight
 *
 * XXX: only do this for the part of runnable > running ?
 *
 */
/*
 * update_tg_cfs_util - 把子队列的利用率变化同步给组实体及父队列。
 */
static inline void
update_tg_cfs_util(struct cfs_rq *cfs_rq, struct sched_entity *se, struct cfs_rq *gcfs_rq)
{
	long delta_sum, delta_avg = gcfs_rq->avg.util_avg - se->avg.util_avg;
	u32 new_sum, divider;

	/* Nothing to update */
	if (!delta_avg)
		return;

	/*
	 * cfs_rq->avg.period_contrib can be used for both cfs_rq and se.
	 * See ___update_load_avg() for details.
	 */
	divider = get_pelt_divider(&cfs_rq->avg);


	/* Set new sched_entity's utilization */
	se->avg.util_avg = gcfs_rq->avg.util_avg;
	new_sum = se->avg.util_avg * divider;
	delta_sum = (long)new_sum - (long)se->avg.util_sum;
	se->avg.util_sum = new_sum;

	/* Update parent cfs_rq utilization */
	add_positive(&cfs_rq->avg.util_avg, delta_avg);
	add_positive(&cfs_rq->avg.util_sum, delta_sum);

	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.util_sum = max_t(u32, cfs_rq->avg.util_sum,
					  cfs_rq->avg.util_avg * PELT_MIN_DIVIDER);
}

/*
 * update_tg_cfs_runnable - 把子队列的可运行需求变化同步给组实体及父队列。
 */
static inline void
update_tg_cfs_runnable(struct cfs_rq *cfs_rq, struct sched_entity *se, struct cfs_rq *gcfs_rq)
{
	long delta_sum, delta_avg = gcfs_rq->avg.runnable_avg - se->avg.runnable_avg;
	u64 new_sum;
	u32 divider;

	/* Nothing to update */
	if (!delta_avg)
		return;

	/*
	 * cfs_rq->avg.period_contrib can be used for both cfs_rq and se.
	 * See ___update_load_avg() for details.
	 */
	divider = get_pelt_divider(&cfs_rq->avg);

	/* Set new sched_entity's runnable */
	se->avg.runnable_avg = gcfs_rq->avg.runnable_avg;
	new_sum = (u64)se->avg.runnable_avg * divider;
	delta_sum = (long)new_sum - (long)se->avg.runnable_sum;
	se->avg.runnable_sum = new_sum;

	/* Update parent cfs_rq runnable */
	add_positive(&cfs_rq->avg.runnable_avg, delta_avg);
	add_positive(&cfs_rq->avg.runnable_sum, delta_sum);
	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.runnable_sum = max_t(u32, cfs_rq->avg.runnable_sum,
					      cfs_rq->avg.runnable_avg * PELT_MIN_DIVIDER);
}

/*
 * update_tg_cfs_load：
 * 将子队列的可运行时间变化折算为组实体加权负载并向上传递。
 */
static inline void
update_tg_cfs_load(struct cfs_rq *cfs_rq, struct sched_entity *se, struct cfs_rq *gcfs_rq)
{
	long delta_avg, running_sum, runnable_sum = gcfs_rq->prop_runnable_sum;
	unsigned long load_avg;
	u64 load_sum = 0;
	s64 delta_sum;
	u32 divider;

	if (!runnable_sum)
		return;

	gcfs_rq->prop_runnable_sum = 0;

	/*
	 * cfs_rq->avg.period_contrib can be used for both cfs_rq and se.
	 * See ___update_load_avg() for details.
	 */
	divider = get_pelt_divider(&cfs_rq->avg);

	if (runnable_sum >= 0) {
		/*
		 * Add runnable; clip at LOAD_AVG_MAX. Reflects that until
		 * the CPU is saturated running == runnable.
		 */
		runnable_sum += se->avg.load_sum;
		runnable_sum = min_t(long, runnable_sum, divider);
	} else {
		/*
		 * Estimate the new unweighted runnable_sum of the gcfs_rq by
		 * assuming all tasks are equally runnable.
		 */
		if (scale_load_down(gcfs_rq->load.weight)) {
			load_sum = div_u64(gcfs_rq->avg.load_sum,
				scale_load_down(gcfs_rq->load.weight));
		}

		/* But make sure to not inflate se's runnable */
		runnable_sum = min(se->avg.load_sum, load_sum);
	}

	/*
	 * runnable_sum can't be lower than running_sum
	 * Rescale running sum to be in the same range as runnable sum
	 * running_sum is in [0 : LOAD_AVG_MAX <<  SCHED_CAPACITY_SHIFT]
	 * runnable_sum is in [0 : LOAD_AVG_MAX]
	 */
	running_sum = se->avg.util_sum >> SCHED_CAPACITY_SHIFT;
	runnable_sum = max(runnable_sum, running_sum);

	load_sum = se_weight(se) * runnable_sum;
	load_avg = div_u64(load_sum, divider);

	delta_avg = load_avg - se->avg.load_avg;
	if (!delta_avg)
		return;

	delta_sum = load_sum - (s64)se_weight(se) * se->avg.load_sum;

	se->avg.load_sum = runnable_sum;
	se->avg.load_avg = load_avg;
	add_positive(&cfs_rq->avg.load_avg, delta_avg);
	add_positive(&cfs_rq->avg.load_sum, delta_sum);
	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.load_sum = max_t(u32, cfs_rq->avg.load_sum,
					  cfs_rq->avg.load_avg * PELT_MIN_DIVIDER);
}

/*
 * add_tg_cfs_propagate - 标记该队列存在待上传的 PELT 变化，
 * 并累积可运行时间增量。
 */
static inline void add_tg_cfs_propagate(struct cfs_rq *cfs_rq, long runnable_sum)
{
	cfs_rq->propagate = 1;
	cfs_rq->prop_runnable_sum += runnable_sum;
}

/* Update task and its cfs_rq load average */
/*
 * propagate_entity_load_avg：
 * 将子队列的利用率、可运行需求和负载传播到上一层。
 * 返回非零表示本次确实处理了待传播状态。
 */
static inline int propagate_entity_load_avg(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq, *gcfs_rq;

	if (entity_is_task(se))
		return 0;

	gcfs_rq = group_cfs_rq(se);
	if (!gcfs_rq->propagate)
		return 0;

	gcfs_rq->propagate = 0;

	cfs_rq = cfs_rq_of(se);

	add_tg_cfs_propagate(cfs_rq, gcfs_rq->prop_runnable_sum);

	update_tg_cfs_util(cfs_rq, se, gcfs_rq);
	update_tg_cfs_runnable(cfs_rq, se, gcfs_rq);
	update_tg_cfs_load(cfs_rq, se, gcfs_rq);

	trace_pelt_cfs_tp(cfs_rq);
	trace_pelt_se_tp(se);

	return 1;
}

/*
 * Check if we need to update the load and the utilization of a blocked
 * group_entity:
 */
/*
 * skip_blocked_update - 组实体已无负载和利用率且子队列无待传播变化时，
 * 跳过阻塞更新。
 */
static inline bool skip_blocked_update(struct sched_entity *se)
{
	struct cfs_rq *gcfs_rq = group_cfs_rq(se);

	/*
	 * If sched_entity still have not zero load or utilization, we have to
	 * decay it:
	 */
	if (se->avg.load_avg || se->avg.util_avg)
		return false;

	/*
	 * If there is a pending propagation, we have to update the load and
	 * the utilization of the sched_entity:
	 */
	if (gcfs_rq->propagate)
		return false;

	/*
	 * Otherwise, the load and the utilization of the sched_entity is
	 * already zero and there is no pending propagation, so it will be a
	 * waste of time to try to decay it:
	 */
	return true;
}

#else /* !CONFIG_FAIR_GROUP_SCHED: */

/*
 * update_tg_load_avg - 未启用组调度时，不维护任务组全局负载。
 */
static inline void update_tg_load_avg(struct cfs_rq *cfs_rq) {}

/*
 * clear_tg_offline_cfs_rqs - 未启用组调度时，无离线组队列贡献需要清理。
 */
static inline void clear_tg_offline_cfs_rqs(struct rq *rq) {}

/*
 * propagate_entity_load_avg - 未启用组调度时，无跨层级 PELT 变化需要传播。
 */
static inline int propagate_entity_load_avg(struct sched_entity *se)
{
	return 0;
}

/*
 * add_tg_cfs_propagate - 未启用组调度时，无需记录向父层传播的增量。
 */
static inline void add_tg_cfs_propagate(struct cfs_rq *cfs_rq, long runnable_sum) {}

#endif /* !CONFIG_FAIR_GROUP_SCHED */

#ifdef CONFIG_NO_HZ_COMMON
/*
 * migrate_se_pelt_lag：
 * 在唤醒迁移前补偿源 CPU 空闲期间未及时推进的实体 PELT 衰减。
 */
static inline void migrate_se_pelt_lag(struct sched_entity *se)
{
	u64 throttled = 0, now, lut;
	struct cfs_rq *cfs_rq;
	struct rq *rq;
	bool is_idle;

	if (load_avg_is_decayed(&se->avg))
		return;

	cfs_rq = cfs_rq_of(se);
	rq = rq_of(cfs_rq);

	rcu_read_lock();
	is_idle = is_idle_task(rcu_dereference(rq->curr));
	rcu_read_unlock();

	/*
	 * The lag estimation comes with a cost we don't want to pay all the
	 * time. Hence, limiting to the case where the source CPU is idle and
	 * we know we are at the greatest risk to have an outdated clock.
	 */
	if (!is_idle)
		return;

	/*
	 * Estimated "now" is: last_update_time + cfs_idle_lag + rq_idle_lag, where:
	 *
	 *   last_update_time (the cfs_rq's last_update_time)
	 *	= cfs_rq_clock_pelt()@cfs_rq_idle
	 *      = rq_clock_pelt()@cfs_rq_idle
	 *        - cfs->throttled_clock_pelt_time@cfs_rq_idle
	 *
	 *   cfs_idle_lag (delta between rq's update and cfs_rq's update)
	 *      = rq_clock_pelt()@rq_idle - rq_clock_pelt()@cfs_rq_idle
	 *
	 *   rq_idle_lag (delta between now and rq's update)
	 *      = sched_clock_cpu() - rq_clock()@rq_idle
	 *
	 * We can then write:
	 *
	 *    now = rq_clock_pelt()@rq_idle - cfs->throttled_clock_pelt_time +
	 *          sched_clock_cpu() - rq_clock()@rq_idle
	 * Where:
	 *      rq_clock_pelt()@rq_idle is rq->clock_pelt_idle
	 *      rq_clock()@rq_idle      is rq->clock_idle
	 *      cfs->throttled_clock_pelt_time@cfs_rq_idle
	 *                              is cfs_rq->throttled_pelt_idle
	 */

#ifdef CONFIG_CFS_BANDWIDTH
	throttled = u64_u32_load(cfs_rq->throttled_pelt_idle);
	/* The clock has been stopped for throttling */
	if (throttled == U64_MAX)
		return;
#endif
	now = u64_u32_load(rq->clock_pelt_idle);
	/*
	 * Paired with _update_idle_rq_clock_pelt(). It ensures at the worst case
	 * is observed the old clock_pelt_idle value and the new clock_idle,
	 * which lead to an underestimation. The opposite would lead to an
	 * overestimation.
	 */
	smp_rmb();
	lut = cfs_rq_last_update_time(cfs_rq);

	now -= throttled;
	if (now < lut)
		/*
		 * cfs_rq->avg.last_update_time is more recent than our
		 * estimation, let's use it.
		 */
		now = lut;
	else
		now += sched_clock_cpu(cpu_of(rq)) - u64_u32_load(rq->clock_idle);

	__update_load_avg_blocked_se(now, se);
}
#else /* !CONFIG_NO_HZ_COMMON: */
/*
 * migrate_se_pelt_lag - 未启用 SMP 时，不存在跨 CPU 迁移导致的 PELT 时间滞后。
 */
static void migrate_se_pelt_lag(struct sched_entity *se) {}
#endif /* !CONFIG_NO_HZ_COMMON */

/*
 * update_cfs_rq_load_avg - 先扣除延后收集的离开实体贡献，
 * 再将队列 PELT 推进到 now。
 */
/**
 * update_cfs_rq_load_avg - update the cfs_rq's load/util averages
 * @now: current time, as per cfs_rq_clock_pelt()
 * @cfs_rq: cfs_rq to update
 *
 * The cfs_rq avg is the direct sum of all its entities (blocked and runnable)
 * avg. The immediate corollary is that all (fair) tasks must be attached.
 *
 * cfs_rq->avg is used for task_h_load() and update_cfs_share() for example.
 *
 * Return: true if the load decayed or we removed load.
 *
 * Since both these conditions indicate a changed cfs_rq->avg.load we should
 * call update_tg_load_avg() when this function returns true.
 */
static inline int
update_cfs_rq_load_avg(u64 now, struct cfs_rq *cfs_rq)
{
	unsigned long removed_load = 0, removed_util = 0, removed_runnable = 0;
	struct sched_avg *sa = &cfs_rq->avg;
	int decayed = 0;

	/*
	 * 跨队列离开时先汇总到 removed，持队列锁更新时再一次性扣除，
	 * 减少锁耦合。
	 */
	if (cfs_rq->removed.nr) {
		unsigned long r;
		u32 divider = get_pelt_divider(&cfs_rq->avg);

		raw_spin_lock(&cfs_rq->removed.lock);
		swap(cfs_rq->removed.util_avg, removed_util);
		swap(cfs_rq->removed.load_avg, removed_load);
		swap(cfs_rq->removed.runnable_avg, removed_runnable);
		cfs_rq->removed.nr = 0;
		raw_spin_unlock(&cfs_rq->removed.lock);

		r = removed_load;
		sub_positive(&sa->load_avg, r);
		sub_positive(&sa->load_sum, r * divider);
		/* See sa->util_sum below */
		sa->load_sum = max_t(u32, sa->load_sum, sa->load_avg * PELT_MIN_DIVIDER);

		r = removed_util;
		sub_positive(&sa->util_avg, r);
		sub_positive(&sa->util_sum, r * divider);
		/*
		 * Because of rounding, se->util_sum might ends up being +1 more than
		 * cfs->util_sum. Although this is not a problem by itself, detaching
		 * a lot of tasks with the rounding problem between 2 updates of
		 * util_avg (~1ms) can make cfs->util_sum becoming null whereas
		 * cfs_util_avg is not.
		 * Check that util_sum is still above its lower bound for the new
		 * util_avg. Given that period_contrib might have moved since the last
		 * sync, we are only sure that util_sum must be above or equal to
		 *    util_avg * minimum possible divider
		 */
		sa->util_sum = max_t(u32, sa->util_sum, sa->util_avg * PELT_MIN_DIVIDER);

		r = removed_runnable;
		sub_positive(&sa->runnable_avg, r);
		sub_positive(&sa->runnable_sum, r * divider);
		/* See sa->util_sum above */
		sa->runnable_sum = max_t(u32, sa->runnable_sum,
					      sa->runnable_avg * PELT_MIN_DIVIDER);

		/*
		 * removed_runnable is the unweighted version of removed_load so we
		 * can use it to estimate removed_load_sum.
		 */
		add_tg_cfs_propagate(cfs_rq,
			-(long)(removed_runnable * divider) >> SCHED_CAPACITY_SHIFT);

		decayed = 1;
	}

	decayed |= __update_load_avg_cfs_rq(now, cfs_rq);
	u64_u32_store_copy(sa->last_update_time,
			   cfs_rq->last_update_time_copy,
			   sa->last_update_time);
	return decayed;
}

/*
 * attach_entity_load_avg - 将实体 PELT 窗口与队列对齐，
 * 再把历史统计接入队列总量。
 */
/**
 * attach_entity_load_avg - attach this entity to its cfs_rq load avg
 * @cfs_rq: cfs_rq to attach to
 * @se: sched_entity to attach
 *
 * Must call update_cfs_rq_load_avg() before this, since we rely on
 * cfs_rq->avg.last_update_time being current.
 */
static void attach_entity_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	/*
	 * cfs_rq->avg.period_contrib can be used for both cfs_rq and se.
	 * See ___update_load_avg() for details.
	 */
	u32 divider = get_pelt_divider(&cfs_rq->avg);

	/*
	 * When we attach the @se to the @cfs_rq, we must align the decay
	 * window because without that, really weird and wonderful things can
	 * happen.
	 *
	 * XXX illustrate
	 */
	se->avg.last_update_time = cfs_rq->avg.last_update_time;
	se->avg.period_contrib = cfs_rq->avg.period_contrib;

	/*
	 * Hell(o) Nasty stuff.. we need to recompute _sum based on the new
	 * period_contrib. This isn't strictly correct, but since we're
	 * entirely outside of the PELT hierarchy, nobody cares if we truncate
	 * _sum a little.
	 */
	se->avg.util_sum = se->avg.util_avg * divider;

	se->avg.runnable_sum = se->avg.runnable_avg * divider;

	se->avg.load_sum = se->avg.load_avg * divider;
	if (se_weight(se) < se->avg.load_sum)
		se->avg.load_sum = div_u64(se->avg.load_sum, se_weight(se));
	else
		se->avg.load_sum = 1;

	enqueue_load_avg(cfs_rq, se);
	cfs_rq->avg.util_avg += se->avg.util_avg;
	cfs_rq->avg.util_sum += se->avg.util_sum;
	cfs_rq->avg.runnable_avg += se->avg.runnable_avg;
	cfs_rq->avg.runnable_sum += se->avg.runnable_sum;

	add_tg_cfs_propagate(cfs_rq, se->avg.load_sum);

	cfs_rq_util_change(cfs_rq, 0);

	trace_pelt_cfs_tp(cfs_rq);
}

/*
 * detach_entity_load_avg - 撤销实体的 PELT 贡献，并通知组层级和调频路径。
 */
/**
 * detach_entity_load_avg - detach this entity from its cfs_rq load avg
 * @cfs_rq: cfs_rq to detach from
 * @se: sched_entity to detach
 *
 * Must call update_cfs_rq_load_avg() before this, since we rely on
 * cfs_rq->avg.last_update_time being current.
 */
static void detach_entity_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	dequeue_load_avg(cfs_rq, se);
	sub_positive(&cfs_rq->avg.util_avg, se->avg.util_avg);
	sub_positive(&cfs_rq->avg.util_sum, se->avg.util_sum);
	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.util_sum = max_t(u32, cfs_rq->avg.util_sum,
					  cfs_rq->avg.util_avg * PELT_MIN_DIVIDER);

	sub_positive(&cfs_rq->avg.runnable_avg, se->avg.runnable_avg);
	sub_positive(&cfs_rq->avg.runnable_sum, se->avg.runnable_sum);
	/* See update_cfs_rq_load_avg() */
	cfs_rq->avg.runnable_sum = max_t(u32, cfs_rq->avg.runnable_sum,
					      cfs_rq->avg.runnable_avg * PELT_MIN_DIVIDER);

	add_tg_cfs_propagate(cfs_rq, -se->avg.load_sum);

	cfs_rq_util_change(cfs_rq, 0);

	trace_pelt_cfs_tp(cfs_rq);
}

/*
 * Optional action to be done while updating the load average
 */
#define UPDATE_TG	0x1
#define SKIP_AGE_LOAD	0x2
#define DO_ATTACH	0x4
#define DO_DETACH	0x8

/* Update task and its cfs_rq load average */
/*
 * update_load_avg - 统一更新实体与队列的 PELT 状态，
 * 并按 flags 接入、分离或传播贡献。
 */
static inline void update_load_avg(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{
	u64 now = cfs_rq_clock_pelt(cfs_rq);
	int decayed;

	/*
	 * Track task load average for carrying it to new CPU after migrated, and
	 * track group sched_entity load average for task_h_load calculation in migration
	 */
	if (se->avg.last_update_time && !(flags & SKIP_AGE_LOAD))
		__update_load_avg_se(now, cfs_rq, se);

	decayed  = update_cfs_rq_load_avg(now, cfs_rq);
	decayed |= propagate_entity_load_avg(se);

	/*
	 * 时间戳为零表示尚未接入当前队列；
	 * 迁移后也用这一标记触发窗口重新对齐。
	 */
	if (!se->avg.last_update_time && (flags & DO_ATTACH)) {

		/*
		 * DO_ATTACH means we're here from enqueue_entity().
		 * !last_update_time means we've passed through
		 * migrate_task_rq_fair() indicating we migrated.
		 *
		 * IOW we're enqueueing a task on a new CPU.
		 */
		attach_entity_load_avg(cfs_rq, se);
		update_tg_load_avg(cfs_rq);

	} else if (flags & DO_DETACH) {
		/*
		 * DO_DETACH means we're here from dequeue_entity()
		 * and we are migrating task out of the CPU.
		 */
		detach_entity_load_avg(cfs_rq, se);
		update_tg_load_avg(cfs_rq);
	} else if (decayed) {
		cfs_rq_util_change(cfs_rq, 0);

		if (flags & UPDATE_TG)
			update_tg_load_avg(cfs_rq);
	}
}

/*
 * Synchronize entity load avg of dequeued entity without locking
 * the previous rq.
 */
/*
 * sync_entity_load_avg：
 * 把阻塞实体的历史负载衰减到所属队列最近一次更新时间。
 */
static void sync_entity_load_avg(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);
	u64 last_update_time;

	last_update_time = cfs_rq_last_update_time(cfs_rq);
	__update_load_avg_blocked_se(last_update_time, se);
}

/*
 * Task first catches up with cfs_rq, and then subtract
 * itself from the cfs_rq (task must be off the queue now).
 */
/*
 * remove_entity_load_avg - 将实体贡献放入 removed 汇总区，
 * 留待队列下次更新时统一扣除。
 */
static void remove_entity_load_avg(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);
	unsigned long flags;

	/*
	 * tasks cannot exit without having gone through wake_up_new_task() ->
	 * enqueue_task_fair() which will have added things to the cfs_rq,
	 * so we can remove unconditionally.
	 */

	sync_entity_load_avg(se);

	raw_spin_lock_irqsave(&cfs_rq->removed.lock, flags);
	++cfs_rq->removed.nr;
	cfs_rq->removed.util_avg	+= se->avg.util_avg;
	cfs_rq->removed.load_avg	+= se->avg.load_avg;
	cfs_rq->removed.runnable_avg	+= se->avg.runnable_avg;
	raw_spin_unlock_irqrestore(&cfs_rq->removed.lock, flags);
}

/*
 * cfs_rq_runnable_avg - 读取队列的平滑可运行需求，包含等待 CPU 的需求。
 */
static inline unsigned long cfs_rq_runnable_avg(struct cfs_rq *cfs_rq)
{
	return cfs_rq->avg.runnable_avg;
}

/*
 * cfs_rq_load_avg - 读取队列按实体权重加权的平滑负载。
 */
static inline unsigned long cfs_rq_load_avg(struct cfs_rq *cfs_rq)
{
	return cfs_rq->avg.load_avg;
}

static int sched_balance_newidle(struct rq *this_rq, struct rq_flags *rf);

/*
 * task_util - 读取任务 PELT 利用率，反映实际获得 CPU 执行的历史比例。
 */
static inline unsigned long task_util(struct task_struct *p)
{
	return READ_ONCE(p->se.avg.util_avg);
}

/*
 * task_runnable - 读取任务可运行需求；竞争 CPU 时它可能高于实际利用率。
 */
static inline unsigned long task_runnable(struct task_struct *p)
{
	return READ_ONCE(p->se.avg.runnable_avg);
}

/*
 * _task_util_est - 读取利用率预测值，并去掉表示样本未变化的状态位。
 */
static inline unsigned long _task_util_est(struct task_struct *p)
{
	return READ_ONCE(p->se.avg.util_est) & ~UTIL_AVG_UNCHANGED;
}

/*
 * task_util_est - 取 PELT 利用率与预测值的较大者，
 * 减少短时负载被低估的情况。
 */
static inline unsigned long task_util_est(struct task_struct *p)
{
	return max(task_util(p), _task_util_est(p));
}

/*
 * util_est_enqueue - 把入队任务的预测利用率加入队列预测总量。
 */
static inline void util_est_enqueue(struct cfs_rq *cfs_rq,
				    struct task_struct *p)
{
	unsigned int enqueued;

	if (!sched_feat(UTIL_EST))
		return;

	/* Update root cfs_rq's estimated utilization */
	enqueued  = cfs_rq->avg.util_est;
	enqueued += _task_util_est(p);
	WRITE_ONCE(cfs_rq->avg.util_est, enqueued);

	trace_sched_util_est_cfs_tp(cfs_rq);
}

/*
 * util_est_dequeue - 从队列预测总量减去离队任务的贡献，并防止下溢。
 */
static inline void util_est_dequeue(struct cfs_rq *cfs_rq,
				    struct task_struct *p)
{
	unsigned int enqueued;

	if (!sched_feat(UTIL_EST))
		return;

	/* Update root cfs_rq's estimated utilization */
	enqueued  = cfs_rq->avg.util_est;
	enqueued -= min_t(unsigned int, enqueued, _task_util_est(p));
	WRITE_ONCE(cfs_rq->avg.util_est, enqueued);

	trace_sched_util_est_cfs_tp(cfs_rq);
}

#define UTIL_EST_MARGIN (SCHED_CAPACITY_SCALE / 100)

/*
 * util_est_update - 在任务睡眠、结束本轮运行后更新利用率预测。
 * 需求上升时快速跟随，下降时平滑衰减，
 * 并排除受竞争影响的不充分样本。
 */
static inline void util_est_update(struct cfs_rq *cfs_rq,
				   struct task_struct *p,
				   bool task_sleep)
{
	unsigned int ewma, dequeued, last_ewma_diff;

	if (!sched_feat(UTIL_EST))
		return;

	/*
	 * Skip update of task's estimated utilization when the task has not
	 * yet completed an activation, e.g. being migrated.
	 */
	if (!task_sleep)
		return;

	/* Get current estimate of utilization */
	ewma = READ_ONCE(p->se.avg.util_est);

	/*
	 * If the PELT values haven't changed since enqueue time,
	 * skip the util_est update.
	 */
	if (ewma & UTIL_AVG_UNCHANGED)
		return;

	/* Get utilization at dequeue */
	dequeued = task_util(p);

	/*
	 * Reset EWMA on utilization increases, the moving average is used only
	 * to smooth utilization decreases.
	 */
	if (ewma <= dequeued) {
		ewma = dequeued;
		goto done;
	}

	/*
	 * Skip update of task's estimated utilization when its members are
	 * already ~1% close to its last activation value.
	 */
	last_ewma_diff = ewma - dequeued;
	if (last_ewma_diff < UTIL_EST_MARGIN)
		goto done;

	/*
	 * To avoid underestimate of task utilization, skip updates of EWMA if
	 * we cannot grant that thread got all CPU time it wanted.
	 */
	/*
	 * 实际利用率明显低于可运行需求，可能是抢不到 CPU，
	 * 不能据此降低需求预测。
	 */
	if ((dequeued + UTIL_EST_MARGIN) < task_runnable(p))
		goto done;


	/*
	 * Update Task's estimated utilization
	 *
	 * When *p completes an activation we can consolidate another sample
	 * of the task size. This is done by using this value to update the
	 * Exponential Weighted Moving Average (EWMA):
	 *
	 *  ewma(t) = w *  task_util(p) + (1-w) * ewma(t-1)
	 *          = w *  task_util(p) +         ewma(t-1)  - w * ewma(t-1)
	 *          = w * (task_util(p) -         ewma(t-1)) +     ewma(t-1)
	 *          = w * (      -last_ewma_diff           ) +     ewma(t-1)
	 *          = w * (-last_ewma_diff +  ewma(t-1) / w)
	 *
	 * Where 'w' is the weight of new samples, which is configured to be
	 * 0.25, thus making w=1/4 ( >>= UTIL_EST_WEIGHT_SHIFT)
	 */
	ewma <<= UTIL_EST_WEIGHT_SHIFT;
	ewma  -= last_ewma_diff;
	ewma >>= UTIL_EST_WEIGHT_SHIFT;
done:
	ewma |= UTIL_AVG_UNCHANGED;
	WRITE_ONCE(p->se.avg.util_est, ewma);

	trace_sched_util_est_se_tp(&p->se);
}

/*
 * get_actual_cpu_capacity - 从架构容量扣除硬件负载或调频压力，
 * 得到当前可用容量估计。
 */
static inline unsigned long get_actual_cpu_capacity(int cpu)
{
	unsigned long capacity = arch_scale_cpu_capacity(cpu);

	capacity -= max(hw_load_avg(cpu_rq(cpu)), cpufreq_get_pressure(cpu));

	return capacity;
}

/*
 * util_fits_cpu - 结合利用率、CPU 容量和 uclamp 限制判断匹配程度。
 * 返回 1 表示匹配，0 表示不匹配，-1 表示容量不足以满足 uclamp_min。
 */
static inline int util_fits_cpu(unsigned long util,
				unsigned long uclamp_min,
				unsigned long uclamp_max,
				int cpu)
{
	unsigned long capacity = capacity_of(cpu);
	unsigned long capacity_orig;
	bool fits, uclamp_max_fits;

	/*
	 * Check if the real util fits without any uclamp boost/cap applied.
	 */
	fits = fits_capacity(util, capacity);

	if (!uclamp_is_used())
		return fits;

	/*
	 * We must use arch_scale_cpu_capacity() for comparing against uclamp_min and
	 * uclamp_max. We only care about capacity pressure (by using
	 * capacity_of()) for comparing against the real util.
	 *
	 * If a task is boosted to 1024 for example, we don't want a tiny
	 * pressure to skew the check whether it fits a CPU or not.
	 *
	 * Similarly if a task is capped to arch_scale_cpu_capacity(little_cpu), it
	 * should fit a little cpu even if there's some pressure.
	 *
	 * Only exception is for HW or cpufreq pressure since it has a direct impact
	 * on available OPP of the system.
	 *
	 * We honour it for uclamp_min only as a drop in performance level
	 * could result in not getting the requested minimum performance level.
	 *
	 * For uclamp_max, we can tolerate a drop in performance level as the
	 * goal is to cap the task. So it's okay if it's getting less.
	 */
	capacity_orig = arch_scale_cpu_capacity(cpu);

	/*
	 * We want to force a task to fit a cpu as implied by uclamp_max.
	 * But we do have some corner cases to cater for..
	 *
	 *
	 *                                 C=z
	 *   |                             ___
	 *   |                  C=y       |   |
	 *   |_ _ _ _ _ _ _ _ _ ___ _ _ _ | _ | _ _ _ _ _  uclamp_max
	 *   |      C=x        |   |      |   |
	 *   |      ___        |   |      |   |
	 *   |     |   |       |   |      |   |    (util somewhere in this region)
	 *   |     |   |       |   |      |   |
	 *   |     |   |       |   |      |   |
	 *   +----------------------------------------
	 *         CPU0        CPU1       CPU2
	 *
	 *   In the above example if a task is capped to a specific performance
	 *   point, y, then when:
	 *
	 *   * util = 80% of x then it does not fit on CPU0 and should migrate
	 *     to CPU1
	 *   * util = 80% of y then it is forced to fit on CPU1 to honour
	 *     uclamp_max request.
	 *
	 *   which is what we're enforcing here. A task always fits if
	 *   uclamp_max <= capacity_orig. But when uclamp_max > capacity_orig,
	 *   the normal upmigration rules should withhold still.
	 *
	 *   Only exception is when we are on max capacity, then we need to be
	 *   careful not to block overutilized state. This is so because:
	 *
	 *     1. There's no concept of capping at max_capacity! We can't go
	 *        beyond this performance level anyway.
	 *     2. The system is being saturated when we're operating near
	 *        max capacity, it doesn't make sense to block overutilized.
	 */
	uclamp_max_fits = (capacity_orig == SCHED_CAPACITY_SCALE) && (uclamp_max == SCHED_CAPACITY_SCALE);
	uclamp_max_fits = !uclamp_max_fits && (uclamp_max <= capacity_orig);
	fits = fits || uclamp_max_fits;

	/*
	 *
	 *                                 C=z
	 *   |                             ___       (region a, capped, util >= uclamp_max)
	 *   |                  C=y       |   |
	 *   |_ _ _ _ _ _ _ _ _ ___ _ _ _ | _ | _ _ _ _ _ uclamp_max
	 *   |      C=x        |   |      |   |
	 *   |      ___        |   |      |   |      (region b, uclamp_min <= util <= uclamp_max)
	 *   |_ _ _|_ _|_ _ _ _| _ | _ _ _| _ | _ _ _ _ _ uclamp_min
	 *   |     |   |       |   |      |   |
	 *   |     |   |       |   |      |   |      (region c, boosted, util < uclamp_min)
	 *   +----------------------------------------
	 *         CPU0        CPU1       CPU2
	 *
	 * a) If util > uclamp_max, then we're capped, we don't care about
	 *    actual fitness value here. We only care if uclamp_max fits
	 *    capacity without taking margin/pressure into account.
	 *    See comment above.
	 *
	 * b) If uclamp_min <= util <= uclamp_max, then the normal
	 *    fits_capacity() rules apply. Except we need to ensure that we
	 *    enforce we remain within uclamp_max, see comment above.
	 *
	 * c) If util < uclamp_min, then we are boosted. Same as (b) but we
	 *    need to take into account the boosted value fits the CPU without
	 *    taking margin/pressure into account.
	 *
	 * Cases (a) and (b) are handled in the 'fits' variable already. We
	 * just need to consider an extra check for case (c) after ensuring we
	 * handle the case uclamp_min > uclamp_max.
	 */
	uclamp_min = min(uclamp_min, uclamp_max);
	/*
	 * 利用率能放下但最低性能要求无法满足时，单独返回 -1，
	 * 供选核路径区分。
	 */
	if (fits && (util < uclamp_min) &&
	    (uclamp_min > get_actual_cpu_capacity(cpu)))
		return -1;

	return fits;
}

/*
 * task_fits_cpu - 按任务预测利用率及有效 uclamp 值检查其是否适合目标 CPU。
 */
static inline int task_fits_cpu(struct task_struct *p, int cpu)
{
	unsigned long uclamp_min = uclamp_eff_value(p, UCLAMP_MIN);
	unsigned long uclamp_max = uclamp_eff_value(p, UCLAMP_MAX);
	unsigned long util = task_util_est(p);
	/*
	 * Return true only if the cpu fully fits the task requirements, which
	 * include the utilization but also the performance hints.
	 */
	return (util_fits_cpu(util, uclamp_min, uclamp_max, cpu) > 0);
}

/*
 * update_misfit_status - 在非对称容量系统标记需要更大 CPU 的任务负载，
 * 供均衡器识别。
 */
static inline void update_misfit_status(struct task_struct *p, struct rq *rq)
{
	int cpu = cpu_of(rq);

	if (!sched_asym_cpucap_active())
		return;

	/*
	 * Affinity allows us to go somewhere higher?  Or are we on biggest
	 * available CPU already? Or do we fit into this CPU ?
	 */
	if (!p || (p->nr_cpus_allowed == 1) ||
	    (arch_scale_cpu_capacity(cpu) == p->max_allowed_capacity) ||
	    task_fits_cpu(p, cpu)) {

		rq->misfit_task_load = 0;
		return;
	}

	/*
	 * Make sure that misfit_task_load will not be null even if
	 * task_h_load() returns 0.
	 */
	rq->misfit_task_load = max_t(unsigned long, task_h_load(p), 1);
}

/*
 * __setparam_fair - 应用 nice 和可选自定义时间片；
 * 自定义请求限制在 0.1 到 100 毫秒。
 */
void __setparam_fair(struct task_struct *p, const struct sched_attr *attr)
{
	struct sched_entity *se = &p->se;

	p->static_prio = NICE_TO_PRIO(attr->sched_nice);
	if (attr->sched_runtime) {
		se->custom_slice = 1;
		se->slice = clamp_t(u64, attr->sched_runtime,
				      NSEC_PER_MSEC/10,   /* HZ=1000 * 10 */
				      NSEC_PER_MSEC*100); /* HZ=100  / 10 */
	} else {
		se->custom_slice = 0;
		se->slice = sysctl_sched_base_slice;
	}
}

/*
 * place_entity - 以队列平均虚拟时间为基准放置实体，
 * 并确定本次虚拟截止时间。
 * 按特性开关保留 lag 或相对截止时间，避免唤醒、迁移改变服务份额。
 */
static void
place_entity(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{
	u64 vslice, vruntime = avg_vruntime(cfs_rq);
	s64 lag = 0;

	if (!se->custom_slice)
		se->slice = sysctl_sched_base_slice;
	vslice = calc_delta_fair(se->slice, se);

	/*
	 * Due to how V is constructed as the weighted average of entities,
	 * adding tasks with positive lag, or removing tasks with negative lag
	 * will move 'time' backwards, this can screw around with the lag of
	 * other tasks.
	 *
	 * EEVDF: placement strategy #1 / #2
	 */
	if (sched_feat(PLACE_LAG) && cfs_rq->nr_queued && se->vlag) {
		struct sched_entity *curr = cfs_rq->curr;
		unsigned long load;

		lag = se->vlag;

		/*
		 * If we want to place a task and preserve lag, we have to
		 * consider the effect of the new entity on the weighted
		 * average and compensate for this, otherwise lag can quickly
		 * evaporate.
		 *
		 * Lag is defined as:
		 *
		 *   lag_i = S - s_i = w_i * (V - v_i)
		 *
		 * To avoid the 'w_i' term all over the place, we only track
		 * the virtual lag:
		 *
		 *   vl_i = V - v_i <=> v_i = V - vl_i
		 *
		 * And we take V to be the weighted average of all v:
		 *
		 *   V = (\Sum w_j*v_j) / W
		 *
		 * Where W is: \Sum w_j
		 *
		 * Then, the weighted average after adding an entity with lag
		 * vl_i is given by:
		 *
		 *   V' = (\Sum w_j*v_j + w_i*v_i) / (W + w_i)
		 *      = (W*V + w_i*(V - vl_i)) / (W + w_i)
		 *      = (W*V + w_i*V - w_i*vl_i) / (W + w_i)
		 *      = (V*(W + w_i) - w_i*vl_i) / (W + w_i)
		 *      = V - w_i*vl_i / (W + w_i)
		 *
		 * And the actual lag after adding an entity with vl_i is:
		 *
		 *   vl'_i = V' - v_i
		 *         = V - w_i*vl_i / (W + w_i) - (V - vl_i)
		 *         = vl_i - w_i*vl_i / (W + w_i)
		 *
		 * Which is strictly less than vl_i. So in order to preserve lag
		 * we should inflate the lag before placement such that the
		 * effective lag after placement comes out right.
		 *
		 * As such, invert the above relation for vl'_i to get the vl_i
		 * we need to use such that the lag after placement is the lag
		 * we computed before dequeue.
		 *
		 *   vl'_i = vl_i - w_i*vl_i / (W + w_i)
		 *         = ((W + w_i)*vl_i - w_i*vl_i) / (W + w_i)
		 *
		 *   (W + w_i)*vl'_i = (W + w_i)*vl_i - w_i*vl_i
		 *                   = W*vl_i
		 *
		 *   vl_i = (W + w_i)*vl'_i / W
		 */
		load = cfs_rq->sum_weight;
		if (curr && curr->on_rq)
			load += scale_load_down(curr->load.weight);

		/*
		 * 新实体加入也会改变加权平均 V，
		 * 预先放大偏移才能保住加入后的 lag。
		 */
		lag *= load + scale_load_down(se->load.weight);
		if (WARN_ON_ONCE(!load))
			load = 1;
		lag = div_s64(lag, load);
	}

	se->vruntime = vruntime - lag;

	/*
	 * 迁移等非睡眠出队可保留相对截止时间，
	 * 在新位置还原后继续原来的请求。
	 */
	if (sched_feat(PLACE_REL_DEADLINE) && se->rel_deadline) {
		se->deadline += se->vruntime;
		se->rel_deadline = 0;
		return;
	}

	/*
	 * When joining the competition; the existing tasks will be,
	 * on average, halfway through their slice, as such start tasks
	 * off with half a slice to ease into the competition.
	 */
	if (sched_feat(PLACE_DEADLINE_INITIAL) && (flags & ENQUEUE_INITIAL))
		vslice /= 2;

	/*
	 * EEVDF: vd_i = ve_i + r_i/w_i
	 */
	se->deadline = se->vruntime + vslice;
}

static void check_enqueue_throttle(struct cfs_rq *cfs_rq);
static inline int cfs_rq_throttled(struct cfs_rq *cfs_rq);

static void
requeue_delayed_entity(struct sched_entity *se);

/*
 * enqueue_entity：
 * 在一层 cfs_rq 上完成运行时间结算、PELT 同步、实体放置及入树。
 */
static void
enqueue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{
	bool curr = cfs_rq->curr == se;

	/*
	 * If we're the current task, we must renormalise before calling
	 * update_curr().
	 */
	if (curr)
		place_entity(cfs_rq, se, flags);

	update_curr(cfs_rq);

	/*
	 * When enqueuing a sched_entity, we must:
	 *   - Update loads to have both entity and cfs_rq synced with now.
	 *   - For group_entity, update its runnable_weight to reflect the new
	 *     h_nr_runnable of its group cfs_rq.
	 *   - For group_entity, update its weight to reflect the new share of
	 *     its group cfs_rq
	 *   - Add its new weight to cfs_rq->load.weight
	 */
	update_load_avg(cfs_rq, se, UPDATE_TG | DO_ATTACH);
	se_update_runnable(se);
	/*
	 * XXX update_load_avg() above will have attached us to the pelt sum;
	 * but update_cfs_group() here will re-adjust the weight and have to
	 * undo/redo all that. Seems wasteful.
	 */
	update_cfs_group(se);

	/*
	 * XXX now that the entity has been re-weighted, and it's lag adjusted,
	 * we can place the entity.
	 */
	if (!curr)
		place_entity(cfs_rq, se, flags);

	account_entity_enqueue(cfs_rq, se);

	/* Entity has migrated, no longer consider this task hot */
	if (flags & ENQUEUE_MIGRATED)
		se->exec_start = 0;

	check_schedstat_required();
	update_stats_enqueue_fair(cfs_rq, se, flags);
	if (!curr)
		__enqueue_entity(cfs_rq, se);
	se->on_rq = 1;

	if (cfs_rq->nr_queued == 1) {
		check_enqueue_throttle(cfs_rq);
		list_add_leaf_cfs_rq(cfs_rq);
#ifdef CONFIG_CFS_BANDWIDTH
		if (cfs_rq->pelt_clock_throttled) {
			struct rq *rq = rq_of(cfs_rq);

			cfs_rq->throttled_clock_pelt_time += rq_clock_pelt(rq) -
				cfs_rq->throttled_clock_pelt;
			cfs_rq->pelt_clock_throttled = 0;
		}
#endif
	}
}

/*
 * __clear_buddies_next - 沿组层级清除仍指向该实体的 next 偏好。
 */
static void __clear_buddies_next(struct sched_entity *se)
{
	for_each_sched_entity(se) {
		struct cfs_rq *cfs_rq = cfs_rq_of(se);
		if (cfs_rq->next != se)
			break;

		cfs_rq->next = NULL;
	}
}

/*
 * clear_buddies - 实体失去继续优先选择的理由时，撤销对应的 buddy 提示。
 */
static void clear_buddies(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	if (cfs_rq->next == se)
		__clear_buddies_next(se);
}

static __always_inline void return_cfs_rq_runtime(struct cfs_rq *cfs_rq);

/*
 * set_delayed - 标记实体延迟出队；任务虽暂留队列，
 * 已不计入层级可运行任务数。
 */
static void set_delayed(struct sched_entity *se)
{
	se->sched_delayed = 1;

	/*
	 * Delayed se of cfs_rq have no tasks queued on them.
	 * Do not adjust h_nr_runnable since dequeue_entities()
	 * will account it for blocked tasks.
	 */
	if (!entity_is_task(se))
		return;

	for_each_sched_entity(se) {
		struct cfs_rq *cfs_rq = cfs_rq_of(se);

		cfs_rq->h_nr_runnable--;
	}
}

/*
 * clear_delayed - 撤销延迟出队标记，并为任务恢复层级可运行计数。
 */
static void clear_delayed(struct sched_entity *se)
{
	se->sched_delayed = 0;

	/*
	 * Delayed se of cfs_rq have no tasks queued on them.
	 * Do not adjust h_nr_runnable since a dequeue has
	 * already accounted for it or an enqueue of a task
	 * below it will account for it in enqueue_task_fair().
	 */
	if (!entity_is_task(se))
		return;

	for_each_sched_entity(se) {
		struct cfs_rq *cfs_rq = cfs_rq_of(se);

		cfs_rq->h_nr_runnable++;
	}
}

/*
 * finish_delayed_dequeue_entity - 结束延迟出队状态，
 * 按 DELAY_ZERO 策略清除正的虚拟 lag。
 */
static inline void finish_delayed_dequeue_entity(struct sched_entity *se)
{
	clear_delayed(se);
	if (sched_feat(DELAY_ZERO) && se->vlag > 0)
		se->vlag = 0;
}

/*
 * dequeue_entity - 在一层队列移除实体并保存公平性状态。
 * 睡眠实体若已超额获得服务，可暂留队列消化负 lag，此时返回 false。
 */
static bool
dequeue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags)
{
	bool sleep = flags & DEQUEUE_SLEEP;
	int action = UPDATE_TG;

	update_curr(cfs_rq);
	clear_buddies(cfs_rq, se);

	if (flags & DEQUEUE_DELAYED) {
		WARN_ON_ONCE(!se->sched_delayed);
	} else {
		bool delay = sleep;
		/*
		 * DELAY_DEQUEUE relies on spurious wakeups, special task
		 * states must not suffer spurious wakeups, excempt them.
		 */
		if (flags & (DEQUEUE_SPECIAL | DEQUEUE_THROTTLE))
			delay = false;

		WARN_ON_ONCE(delay && se->sched_delayed);

		/*
		 * 负 lag 表示已超额获得服务。
		 * 暂留实体让其随队列虚拟时间推进恢复资格，
		 * 避免通过反复短睡眠消除这部分欠账；
		 * 此时不再计入可运行任务数。
		 */
		if (sched_feat(DELAY_DEQUEUE) && delay &&
		    !entity_eligible(cfs_rq, se)) {
			update_load_avg(cfs_rq, se, 0);
			set_delayed(se);
			return false;
		}
	}

	if (entity_is_task(se) && task_on_rq_migrating(task_of(se)))
		action |= DO_DETACH;

	/*
	 * When dequeuing a sched_entity, we must:
	 *   - Update loads to have both entity and cfs_rq synced with now.
	 *   - For group_entity, update its runnable_weight to reflect the new
	 *     h_nr_runnable of its group cfs_rq.
	 *   - Subtract its previous weight from cfs_rq->load.weight.
	 *   - For group entity, update its weight to reflect the new share
	 *     of its group cfs_rq.
	 */
	update_load_avg(cfs_rq, se, action);
	se_update_runnable(se);

	update_stats_dequeue_fair(cfs_rq, se, flags);

	update_entity_lag(cfs_rq, se);
	if (sched_feat(PLACE_REL_DEADLINE) && !sleep) {
		se->deadline -= se->vruntime;
		se->rel_deadline = 1;
	}

	if (se != cfs_rq->curr)
		__dequeue_entity(cfs_rq, se);
	se->on_rq = 0;
	account_entity_dequeue(cfs_rq, se);

	/* return excess runtime on last dequeue */
	return_cfs_rq_runtime(cfs_rq);

	update_cfs_group(se);

	if (flags & DEQUEUE_DELAYED)
		finish_delayed_dequeue_entity(se);

	if (cfs_rq->nr_queued == 0) {
		update_idle_cfs_rq_clock_pelt(cfs_rq);
#ifdef CONFIG_CFS_BANDWIDTH
		if (throttled_hierarchy(cfs_rq)) {
			struct rq *rq = rq_of(cfs_rq);

			list_del_leaf_cfs_rq(cfs_rq);
			cfs_rq->throttled_clock_pelt = rq_clock_pelt(rq);
			cfs_rq->pelt_clock_throttled = 1;
		}
#endif
	}

	return true;
}

/*
 * set_next_entity - 把选中的实体从红黑树转为当前实体，
 * 并设置执行起点及保护区间。
 */
static void
set_next_entity(struct cfs_rq *cfs_rq, struct sched_entity *se, bool first)
{
	clear_buddies(cfs_rq, se);

	/* 'current' is not kept within the tree. */
	/*
	 * 成为当前实体只移出时间线树，on_rq 仍为真，
	 * 队列成员身份继续保留。
	 */
	if (se->on_rq) {
		/*
		 * Any task has to be enqueued before it get to execute on
		 * a CPU. So account for the time it spent waiting on the
		 * runqueue.
		 */
		update_stats_wait_end_fair(cfs_rq, se);
		__dequeue_entity(cfs_rq, se);
		update_load_avg(cfs_rq, se, UPDATE_TG);

		if (first)
			set_protect_slice(cfs_rq, se);
	}

	update_stats_curr_start(cfs_rq, se);
	WARN_ON_ONCE(cfs_rq->curr);
	cfs_rq->curr = se;

	/*
	 * Track our maximum slice length, if the CPU's load is at
	 * least twice that of our own weight (i.e. don't track it
	 * when there are only lesser-weight tasks around):
	 */
	if (schedstat_enabled() &&
	    rq_of(cfs_rq)->cfs.load.weight >= 2*se->load.weight) {
		struct sched_statistics *stats;

		stats = __schedstats_from_se(se);
		__schedstat_set(stats->slice_max,
				max((u64)stats->slice_max,
				    se->sum_exec_runtime - se->prev_sum_exec_runtime));
	}

	se->prev_sum_exec_runtime = se->sum_exec_runtime;
}

static int dequeue_entities(struct rq *rq, struct sched_entity *se, int flags);

/*
 * Pick the next process, keeping these things in mind, in this order:
 * 1) keep things fair between processes/task groups
 * 2) pick the "next" process, since someone really wants that to run
 * 3) pick the "last" process, for cache locality
 * 4) do not run the "skip" process, if something else is available
 */
/*
 * pick_next_entity - 取得 EEVDF 候选；若候选是延迟出队实体，
 * 则完成出队并返回 NULL。
 */
static struct sched_entity *
pick_next_entity(struct rq *rq, struct cfs_rq *cfs_rq, bool protect)
{
	struct sched_entity *se;

	se = pick_eevdf(cfs_rq, protect);
	if (se->sched_delayed) {
		/*
		 * 被选中的是等待完成出队的实体，
		 * 需完成阻塞流程并重新选取，不能交给 CPU 执行。
		 */
		dequeue_entities(rq, se, DEQUEUE_SLEEP | DEQUEUE_DELAYED);
		/*
		 * Must not reference @se again, see __block_task().
		 */
		return NULL;
	}
	return se;
}

static bool check_cfs_rq_runtime(struct cfs_rq *cfs_rq);

/*
 * put_prev_entity - 结算上一实体并检查带宽，
 * 将仍在队的实体放回红黑树等待竞争。
 */
static void put_prev_entity(struct cfs_rq *cfs_rq, struct sched_entity *prev)
{
	/*
	 * If still on the runqueue then deactivate_task()
	 * was not called and update_curr() has to be done:
	 */
	if (prev->on_rq)
		update_curr(cfs_rq);

	/* throttle cfs_rqs exceeding runtime */
	check_cfs_rq_runtime(cfs_rq);

	if (prev->on_rq) {
		update_stats_wait_start_fair(cfs_rq, prev);
		/* Put 'current' back into the tree. */
		__enqueue_entity(cfs_rq, prev);
		/* in !on_rq case, update occurred at dequeue */
		update_load_avg(cfs_rq, prev, 0);
	}
	WARN_ON_ONCE(cfs_rq->curr != prev);
	cfs_rq->curr = NULL;
}

/*
 * entity_tick - 在时钟 tick 中结算实体运行和负载；
 * 高精度时间片到期时请求重调度。
 */
static void
entity_tick(struct cfs_rq *cfs_rq, struct sched_entity *curr, int queued)
{
	/*
	 * Update run-time statistics of the 'current'.
	 */
	update_curr(cfs_rq);

	/*
	 * Ensure that runnable average is periodically updated.
	 */
	update_load_avg(cfs_rq, curr, UPDATE_TG);
	update_cfs_group(curr);

#ifdef CONFIG_SCHED_HRTICK
	/*
	 * queued ticks are scheduled to match the slice, so don't bother
	 * validating it and just reschedule.
	 */
	if (queued) {
		resched_curr_lazy(rq_of(cfs_rq));
		return;
	}
#endif
}


/**************************************************
 * CFS bandwidth control machinery
 */

/*
 * 带宽控制使用任务组共享配额池和每 CPU 本地余额两层结构。
 * quota/period 限制组的 CPU 时间，EEVDF slice 决定单次公平请求长度；
 * 本地余额不足时先从共享池补充，共享池也不足才进入限流。
 */
#ifdef CONFIG_CFS_BANDWIDTH

#ifdef CONFIG_JUMP_LABEL
static struct static_key __cfs_bandwidth_used;

/*
 * cfs_bandwidth_used：
 * 通过静态分支判断系统中是否有任务组正在使用带宽限制。
 */
static inline bool cfs_bandwidth_used(void)
{
	return static_key_false(&__cfs_bandwidth_used);
}

/*
 * cfs_bandwidth_usage_inc - 增加带宽控制使用计数，
 * 必要时启用相关快速路径分支。
 */
void cfs_bandwidth_usage_inc(void)
{
	static_key_slow_inc_cpuslocked(&__cfs_bandwidth_used);
}

/*
 * cfs_bandwidth_usage_dec - 减少带宽控制使用计数，
 * 无使用者时关闭相关快速路径分支。
 */
void cfs_bandwidth_usage_dec(void)
{
	static_key_slow_dec_cpuslocked(&__cfs_bandwidth_used);
}
#else /* !CONFIG_JUMP_LABEL: */
/*
 * cfs_bandwidth_used - 没有跳转标签优化时始终进入带宽检查，
 * 再由队列状态决定是否记账。
 */
static bool cfs_bandwidth_used(void)
{
	return true;
}

/*
 * cfs_bandwidth_usage_inc - 没有跳转标签优化时，无需维护静态分支引用计数。
 */
void cfs_bandwidth_usage_inc(void) {}
/*
 * cfs_bandwidth_usage_dec - 没有跳转标签优化时，无需更新静态分支。
 */
void cfs_bandwidth_usage_dec(void) {}
#endif /* !CONFIG_JUMP_LABEL */

/*
 * sched_cfs_bandwidth_slice：
 * 将每次从组配额池分配的运行时间由微秒换算为纳秒。
 */
static inline u64 sched_cfs_bandwidth_slice(void)
{
	return (u64)sysctl_sched_cfs_bandwidth_slice * NSEC_PER_USEC;
}

/*
 * Replenish runtime according to assigned quota. We use sched_clock_cpu
 * directly instead of rq->clock to avoid adding additional synchronization
 * around rq->lock.
 *
 * requires cfs_b->lock
 */
/*
 * __refill_cfs_bandwidth_runtime - 补充一个周期的组配额，
 * 并将可积累时间限制为 quota + burst。
 */
void __refill_cfs_bandwidth_runtime(struct cfs_bandwidth *cfs_b)
{
	s64 runtime;

	if (unlikely(cfs_b->quota == RUNTIME_INF))
		return;

	cfs_b->runtime += cfs_b->quota;
	runtime = cfs_b->runtime_snap - cfs_b->runtime;
	if (runtime > 0) {
		cfs_b->burst_time += runtime;
		cfs_b->nr_burst++;
	}

	cfs_b->runtime = min(cfs_b->runtime, cfs_b->quota + cfs_b->burst);
	cfs_b->runtime_snap = cfs_b->runtime;
}

/*
 * tg_cfs_bandwidth - 取得任务组共享的带宽配额池。
 */
static inline struct cfs_bandwidth *tg_cfs_bandwidth(struct task_group *tg)
{
	return &tg->cfs_bandwidth;
}

/* returns 0 on failure to allocate runtime */
/*
 * __assign_cfs_rq_runtime - 持有带宽锁时，
 * 尝试从组配额池给本地队列补充运行时间。
 * 返回非零表示补充后本地余额为正，不保证已达到请求的目标值。
 */
static int __assign_cfs_rq_runtime(struct cfs_bandwidth *cfs_b,
				   struct cfs_rq *cfs_rq, u64 target_runtime)
{
	u64 min_amount, amount = 0;

	lockdep_assert_held(&cfs_b->lock);

	/* note: this is a positive sum as runtime_remaining <= 0 */
	min_amount = target_runtime - cfs_rq->runtime_remaining;

	if (cfs_b->quota == RUNTIME_INF)
		amount = min_amount;
	else {
		start_cfs_bandwidth(cfs_b);

		if (cfs_b->runtime > 0) {
			amount = min(cfs_b->runtime, min_amount);
			cfs_b->runtime -= amount;
			cfs_b->idle = 0;
		}
	}

	cfs_rq->runtime_remaining += amount;

	return cfs_rq->runtime_remaining > 0;
}

/* returns 0 on failure to allocate runtime */
/*
 * assign_cfs_rq_runtime - 加锁后按标准分配粒度为本地队列申请运行时间。
 */
static int assign_cfs_rq_runtime(struct cfs_rq *cfs_rq)
{
	struct cfs_bandwidth *cfs_b = tg_cfs_bandwidth(cfs_rq->tg);
	int ret;

	raw_spin_lock(&cfs_b->lock);
	ret = __assign_cfs_rq_runtime(cfs_b, cfs_rq, sched_cfs_bandwidth_slice());
	raw_spin_unlock(&cfs_b->lock);

	return ret;
}

/*
 * __account_cfs_rq_runtime - 扣除本地已运行时间，耗尽后申请配额；
 * 申请失败则请求重调度。
 */
static void __account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec)
{
	/* dock delta_exec before expiring quota (as it could span periods) */
	cfs_rq->runtime_remaining -= delta_exec;

	if (likely(cfs_rq->runtime_remaining > 0))
		return;

	if (cfs_rq->throttled)
		return;
	/*
	 * if we're unable to extend our runtime we resched so that the active
	 * hierarchy can be throttled
	 */
	if (!assign_cfs_rq_runtime(cfs_rq) && likely(cfs_rq->curr))
		resched_curr(rq_of(cfs_rq));
}

/*
 * account_cfs_rq_runtime - 仅对实际启用带宽限制的队列执行配额记账。
 */
static __always_inline
void account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec)
{
	if (!cfs_bandwidth_used() || !cfs_rq->runtime_enabled)
		return;

	__account_cfs_rq_runtime(cfs_rq, delta_exec);
}

/*
 * cfs_rq_throttled - 判断本队列是否因自身配额耗尽而处于限流状态。
 */
static inline int cfs_rq_throttled(struct cfs_rq *cfs_rq)
{
	return cfs_bandwidth_used() && cfs_rq->throttled;
}

/*
 * cfs_rq_pelt_clock_throttled - 判断限流是否已暂停该队列的 PELT 时钟。
 */
static inline bool cfs_rq_pelt_clock_throttled(struct cfs_rq *cfs_rq)
{
	return cfs_bandwidth_used() && cfs_rq->pelt_clock_throttled;
}

/* check whether cfs_rq, or any parent, is throttled */
/*
 * throttled_hierarchy - 检查本队列或祖先队列是否存在尚未解除的限流。
 */
static inline int throttled_hierarchy(struct cfs_rq *cfs_rq)
{
	return cfs_bandwidth_used() && cfs_rq->throttle_count;
}

/*
 * lb_throttled_hierarchy - 检查任务所在组在目标 CPU 上是否受限流约束。
 */
static inline int lb_throttled_hierarchy(struct task_struct *p, int dst_cpu)
{
	return throttled_hierarchy(task_group(p)->cfs_rq[dst_cpu]);
}

/*
 * task_is_throttled - 判断任务是否已从公平队列转入限流等待状态。
 */
static inline bool task_is_throttled(struct task_struct *p)
{
	return cfs_bandwidth_used() && p->throttled;
}

static bool dequeue_task_fair(struct rq *rq, struct task_struct *p, int flags);
/*
 * throttle_cfs_rq_work - 在任务上下文重新确认层级限流状态，
 * 并把任务移到限流等待链表。
 * 通过 task_work 在返回用户态前执行，避免直接截断正在执行的内核路径。
 */
static void throttle_cfs_rq_work(struct callback_head *work)
{
	struct task_struct *p = container_of(work, struct task_struct, sched_throttle_work);
	struct sched_entity *se;
	struct cfs_rq *cfs_rq;
	struct rq *rq;

	WARN_ON_ONCE(p != current);
	p->sched_throttle_work.next = &p->sched_throttle_work;

	/*
	 * If task is exiting, then there won't be a return to userspace, so we
	 * don't have to bother with any of this.
	 */
	if ((p->flags & PF_EXITING))
		return;

	scoped_guard(task_rq_lock, p) {
		se = &p->se;
		cfs_rq = cfs_rq_of(se);

		/* Raced, forget */
		if (p->sched_class != &fair_sched_class)
			return;

		/*
		 * If not in limbo, then either replenish has happened or this
		 * task got migrated out of the throttled cfs_rq, move along.
		 */
		if (!cfs_rq->throttle_count)
			return;
		rq = scope.rq;
		update_rq_clock(rq);
		WARN_ON_ONCE(p->throttled || !list_empty(&p->throttle_node));
		dequeue_task_fair(rq, p, DEQUEUE_SLEEP | DEQUEUE_THROTTLE);
		/*
		 * 离开公平队列后暂存到专用链表，
		 * 解除全部层级限流时由 tg_unthrottle_up() 恢复。
		 */
		list_add(&p->throttle_node, &cfs_rq->throttled_limbo_list);
		/*
		 * Must not set throttled before dequeue or dequeue will
		 * mistakenly regard this task as an already throttled one.
		 */
		p->throttled = true;
		resched_curr(rq);
	}
}

/*
 * init_cfs_throttle_work - 初始化任务的限流回调、未排队标记和等待链表节点。
 */
void init_cfs_throttle_work(struct task_struct *p)
{
	init_task_work(&p->sched_throttle_work, throttle_cfs_rq_work);
	/* Protect against double add, see throttle_cfs_rq() and throttle_cfs_rq_work() */
	p->sched_throttle_work.next = &p->sched_throttle_work;
	INIT_LIST_HEAD(&p->throttle_node);
}

/*
 * Task is throttled and someone wants to dequeue it again:
 * it could be sched/core when core needs to do things like
 * task affinity change, task group change, task sched class
 * change etc. and in these cases, DEQUEUE_SLEEP is not set;
 * or the task is blocked after throttled due to freezer etc.
 * and in these cases, DEQUEUE_SLEEP is set.
 */
static void detach_task_cfs_rq(struct task_struct *p);
/*
 * dequeue_throttled_task - 移除限流等待任务，
 * 按睡眠或迁移原因清理状态及负载关系。
 */
static void dequeue_throttled_task(struct task_struct *p, int flags)
{
	WARN_ON_ONCE(p->se.on_rq);
	list_del_init(&p->throttle_node);

	/* task blocked after throttled */
	if (flags & DEQUEUE_SLEEP) {
		p->throttled = false;
		return;
	}

	/*
	 * task is migrating off its old cfs_rq, detach
	 * the task's load from its old cfs_rq.
	 */
	if (task_on_rq_migrating(p))
		detach_task_cfs_rq(p);
}

/*
 * enqueue_throttled_task - 重新入队时检查层级限流；
 * 仍受限则留在等待链表并返回 true。
 */
static bool enqueue_throttled_task(struct task_struct *p)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(&p->se);

	/* @p should have gone through dequeue_throttled_task() first */
	WARN_ON_ONCE(!list_empty(&p->throttle_node));

	/*
	 * If the throttled task @p is enqueued to a throttled cfs_rq,
	 * take the fast path by directly putting the task on the
	 * target cfs_rq's limbo list.
	 *
	 * Do not do that when @p is current because the following race can
	 * cause @p's group_node to be incorectly re-insterted in its rq's
	 * cfs_tasks list, despite being throttled:
	 *
	 *     cpuX                       cpuY
	 *   p ret2user
	 *  throttle_cfs_rq_work()  sched_move_task(p)
	 *  LOCK task_rq_lock
	 *  dequeue_task_fair(p)
	 *  UNLOCK task_rq_lock
	 *                          LOCK task_rq_lock
	 *                          task_current_donor(p) == true
	 *                          task_on_rq_queued(p) == true
	 *                          dequeue_task(p)
	 *                          put_prev_task(p)
	 *                          sched_change_group()
	 *                          enqueue_task(p) -> p's new cfs_rq
	 *                                             is throttled, go
	 *                                             fast path and skip
	 *                                             actual enqueue
	 *                          set_next_task(p)
	 *                    list_move(&se->group_node, &rq->cfs_tasks); // bug
	 *  schedule()
	 *
	 * In the above race case, @p current cfs_rq is in the same rq as
	 * its previous cfs_rq because sched_move_task() only moves a task
	 * to a different group from the same rq, so we can use its current
	 * cfs_rq to derive rq and test if the task is current.
	 */
	if (throttled_hierarchy(cfs_rq) &&
	    !task_current_donor(rq_of(cfs_rq), p)) {
		list_add(&p->throttle_node, &cfs_rq->throttled_limbo_list);
		return true;
	}

	/* we can't take the fast path, do an actual enqueue*/
	p->throttled = false;
	return false;
}

static void enqueue_task_fair(struct rq *rq, struct task_struct *p, int flags);
/*
 * tg_unthrottle_up - 自底向上解除一层限流计数；
 * 全部解除后恢复时钟并重新入队等待任务。
 */
static int tg_unthrottle_up(struct task_group *tg, void *data)
{
	struct rq *rq = data;
	struct cfs_rq *cfs_rq = tg->cfs_rq[cpu_of(rq)];
	struct task_struct *p, *tmp;

	if (--cfs_rq->throttle_count)
		return 0;

	if (cfs_rq->pelt_clock_throttled) {
		cfs_rq->throttled_clock_pelt_time += rq_clock_pelt(rq) -
					     cfs_rq->throttled_clock_pelt;
		cfs_rq->pelt_clock_throttled = 0;
	}

	if (cfs_rq->throttled_clock_self) {
		u64 delta = rq_clock(rq) - cfs_rq->throttled_clock_self;

		cfs_rq->throttled_clock_self = 0;

		if (WARN_ON_ONCE((s64)delta < 0))
			delta = 0;

		cfs_rq->throttled_clock_self_time += delta;
	}

	/* Re-enqueue the tasks that have been throttled at this level. */
	list_for_each_entry_safe(p, tmp, &cfs_rq->throttled_limbo_list, throttle_node) {
		list_del_init(&p->throttle_node);
		p->throttled = false;
		enqueue_task_fair(rq_of(cfs_rq), p, ENQUEUE_WAKEUP);
	}

	/* Add cfs_rq with load or one or more already running entities to the list */
	if (!cfs_rq_is_decayed(cfs_rq))
		list_add_leaf_cfs_rq(cfs_rq);

	return 0;
}

/*
 * task_has_throttle_work - 检查任务是否已有待执行的限流回调，避免重复提交。
 */
static inline bool task_has_throttle_work(struct task_struct *p)
{
	return p->sched_throttle_work.next != &p->sched_throttle_work;
}

/*
 * task_throttle_setup_work：
 * 为可处理的用户任务安排返回用户态前执行的限流工作。
 */
static inline void task_throttle_setup_work(struct task_struct *p)
{
	if (task_has_throttle_work(p))
		return;

	/*
	 * Kthreads and exiting tasks don't return to userspace, so adding the
	 * work is pointless
	 */
	if ((p->flags & (PF_EXITING | PF_KTHREAD)))
		return;

	task_work_add(p, &p->sched_throttle_work, TWA_RESUME);
}

/*
 * record_throttle_clock - 在任务因限流离队时记录队列及层级限流统计起点。
 */
static void record_throttle_clock(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);

	if (cfs_rq_throttled(cfs_rq) && !cfs_rq->throttled_clock)
		cfs_rq->throttled_clock = rq_clock(rq);

	if (!cfs_rq->throttled_clock_self)
		cfs_rq->throttled_clock_self = rq_clock(rq);
}

/*
 * tg_throttle_down - 向子组传播限流计数，
 * 并暂停已无排队实体的队列 PELT 时钟。
 */
static int tg_throttle_down(struct task_group *tg, void *data)
{
	struct rq *rq = data;
	struct cfs_rq *cfs_rq = tg->cfs_rq[cpu_of(rq)];

	if (cfs_rq->throttle_count++)
		return 0;

	/*
	 * For cfs_rqs that still have entities enqueued, PELT clock
	 * stop happens at dequeue time when all entities are dequeued.
	 */
	if (!cfs_rq->nr_queued) {
		list_del_leaf_cfs_rq(cfs_rq);
		cfs_rq->throttled_clock_pelt = rq_clock_pelt(rq);
		cfs_rq->pelt_clock_throttled = 1;
	}

	WARN_ON_ONCE(cfs_rq->throttled_clock_self);
	WARN_ON_ONCE(!list_empty(&cfs_rq->throttled_limbo_list));
	return 0;
}

/*
 * throttle_cfs_rq - 再次尝试获取配额，
 * 仍失败才把队列加入限流链表并向下传播状态。
 * 此处标记限流，任务实际离队由后续 task_work 路径完成。
 */
static bool throttle_cfs_rq(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	struct cfs_bandwidth *cfs_b = tg_cfs_bandwidth(cfs_rq->tg);
	int dequeue = 1;

	raw_spin_lock(&cfs_b->lock);
	/* This will start the period timer if necessary */
	/*
	 * 在锁内再检查一次，
	 * 处理周期补充或其他 CPU 归还配额与本次限流之间的竞争。
	 */
	if (__assign_cfs_rq_runtime(cfs_b, cfs_rq, 1)) {
		/*
		 * We have raced with bandwidth becoming available, and if we
		 * actually throttled the timer might not unthrottle us for an
		 * entire period. We additionally needed to make sure that any
		 * subsequent check_cfs_rq_runtime calls agree not to throttle
		 * us, as we may commit to do cfs put_prev+pick_next, so we ask
		 * for 1ns of runtime rather than just check cfs_b.
		 */
		dequeue = 0;
	} else {
		list_add_tail_rcu(&cfs_rq->throttled_list,
				  &cfs_b->throttled_cfs_rq);
	}
	raw_spin_unlock(&cfs_b->lock);

	if (!dequeue)
		return false;  /* Throttle no longer required. */

	/* freeze hierarchy runnable averages while throttled */
	rcu_read_lock();
	walk_tg_tree_from(cfs_rq->tg, tg_throttle_down, tg_nop, (void *)rq);
	rcu_read_unlock();

	/*
	 * Note: distribution will already see us throttled via the
	 * throttled-list.  rq->lock protects completion.
	 */
	cfs_rq->throttled = 1;
	WARN_ON_ONCE(cfs_rq->throttled_clock);
	return true;
}

/*
 * unthrottle_cfs_rq - 配额可用后解除队列及后代的相应限流，
 * 恢复等待任务和叶子链表。
 */
void unthrottle_cfs_rq(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	struct cfs_bandwidth *cfs_b = tg_cfs_bandwidth(cfs_rq->tg);
	struct sched_entity *se = cfs_rq->tg->se[cpu_of(rq)];

	/*
	 * It's possible we are called with runtime_remaining < 0 due to things
	 * like async unthrottled us with a positive runtime_remaining but other
	 * still running entities consumed those runtime before we reached here.
	 *
	 * We can't unthrottle this cfs_rq without any runtime remaining because
	 * any enqueue in tg_unthrottle_up() will immediately trigger a throttle,
	 * which is not supposed to happen on unthrottle path.
	 */
	if (cfs_rq->runtime_enabled && cfs_rq->runtime_remaining <= 0)
		return;

	cfs_rq->throttled = 0;

	update_rq_clock(rq);

	raw_spin_lock(&cfs_b->lock);
	if (cfs_rq->throttled_clock) {
		cfs_b->throttled_time += rq_clock(rq) - cfs_rq->throttled_clock;
		cfs_rq->throttled_clock = 0;
	}
	list_del_rcu(&cfs_rq->throttled_list);
	raw_spin_unlock(&cfs_b->lock);

	/* update hierarchical throttle state */
	walk_tg_tree_from(cfs_rq->tg, tg_nop, tg_unthrottle_up, (void *)rq);

	if (!cfs_rq->load.weight) {
		if (!cfs_rq->on_list)
			return;
		/*
		 * Nothing to run but something to decay (on_list)?
		 * Complete the branch.
		 */
		for_each_sched_entity(se) {
			if (list_add_leaf_cfs_rq(cfs_rq_of(se)))
				break;
		}
	}

	assert_list_leaf_cfs_rq(rq);

	/* Determine whether we need to wake up potentially idle CPU: */
	if (rq->curr == rq->idle && rq->cfs.nr_queued)
		resched_curr(rq);
}

/*
 * __cfsb_csd_unthrottle：
 * 在运行队列锁保护下批量处理该 CPU 待解除限流的队列。
 */
static void __cfsb_csd_unthrottle(void *arg)
{
	struct cfs_rq *cursor, *tmp;
	struct rq *rq = arg;
	struct rq_flags rf;

	rq_lock(rq, &rf);

	/*
	 * Iterating over the list can trigger several call to
	 * update_rq_clock() in unthrottle_cfs_rq().
	 * Do it once and skip the potential next ones.
	 */
	update_rq_clock(rq);
	rq_clock_start_loop_update(rq);

	/*
	 * Since we hold rq lock we're safe from concurrent manipulation of
	 * the CSD list. However, this RCU critical section annotates the
	 * fact that we pair with sched_free_group_rcu(), so that we cannot
	 * race with group being freed in the window between removing it
	 * from the list and advancing to the next entry in the list.
	 */
	rcu_read_lock();

	list_for_each_entry_safe(cursor, tmp, &rq->cfsb_csd_list,
				 throttled_csd_list) {
		list_del_init(&cursor->throttled_csd_list);

		if (cfs_rq_throttled(cursor))
			unthrottle_cfs_rq(cursor);
	}

	rcu_read_unlock();

	rq_clock_stop_loop_update(rq);
	rq_unlock(rq, &rf);
}

/*
 * __unthrottle_cfs_rq_async - 本地队列直接解限，
 * 远端队列加入批处理链表并按需发送异步调用。
 */
static inline void __unthrottle_cfs_rq_async(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	bool first;

	if (rq == this_rq()) {
		unthrottle_cfs_rq(cfs_rq);
		return;
	}

	/* Already enqueued */
	if (WARN_ON_ONCE(!list_empty(&cfs_rq->throttled_csd_list)))
		return;

	first = list_empty(&rq->cfsb_csd_list);
	list_add_tail(&cfs_rq->throttled_csd_list, &rq->cfsb_csd_list);
	if (first)
		smp_call_function_single_async(cpu_of(rq), &rq->cfsb_csd);
}

/*
 * unthrottle_cfs_rq_async - 检查限流状态和正配额余额，再安排异步解除限流。
 */
static void unthrottle_cfs_rq_async(struct cfs_rq *cfs_rq)
{
	lockdep_assert_rq_held(rq_of(cfs_rq));

	if (WARN_ON_ONCE(!cfs_rq_throttled(cfs_rq) ||
	    cfs_rq->runtime_remaining <= 0))
		return;

	__unthrottle_cfs_rq_async(cfs_rq);
}

/*
 * distribute_cfs_runtime - 把全局剩余配额分给受限队列，并安排恢复运行。
 * 先补足欠账使余额转正；返回值表示是否仍有队列因配额不足而受限。
 */
static bool distribute_cfs_runtime(struct cfs_bandwidth *cfs_b)
{
	int this_cpu = smp_processor_id();
	u64 runtime, remaining = 1;
	bool throttled = false;
	struct cfs_rq *cfs_rq, *tmp;
	struct rq_flags rf;
	struct rq *rq;
	LIST_HEAD(local_unthrottle);

	rcu_read_lock();
	list_for_each_entry_rcu(cfs_rq, &cfs_b->throttled_cfs_rq,
				throttled_list) {
		rq = rq_of(cfs_rq);

		if (!remaining) {
			throttled = true;
			break;
		}

		rq_lock_irqsave(rq, &rf);
		if (!cfs_rq_throttled(cfs_rq))
			goto next;

		/* Already queued for async unthrottle */
		if (!list_empty(&cfs_rq->throttled_csd_list))
			goto next;

		/* By the above checks, this should never be true */
		WARN_ON_ONCE(cfs_rq->runtime_remaining > 0);

		raw_spin_lock(&cfs_b->lock);
		runtime = -cfs_rq->runtime_remaining + 1;
		if (runtime > cfs_b->runtime)
			runtime = cfs_b->runtime;
		cfs_b->runtime -= runtime;
		remaining = cfs_b->runtime;
		raw_spin_unlock(&cfs_b->lock);

		cfs_rq->runtime_remaining += runtime;

		/* we check whether we're throttled above */
		if (cfs_rq->runtime_remaining > 0) {
			if (cpu_of(rq) != this_cpu) {
				unthrottle_cfs_rq_async(cfs_rq);
			} else {
				/*
				 * We currently only expect to be unthrottling
				 * a single cfs_rq locally.
				 */
				WARN_ON_ONCE(!list_empty(&local_unthrottle));
				list_add_tail(&cfs_rq->throttled_csd_list,
					      &local_unthrottle);
			}
		} else {
			throttled = true;
		}

next:
		rq_unlock_irqrestore(rq, &rf);
	}

	list_for_each_entry_safe(cfs_rq, tmp, &local_unthrottle,
				 throttled_csd_list) {
		struct rq *rq = rq_of(cfs_rq);

		rq_lock_irqsave(rq, &rf);

		list_del_init(&cfs_rq->throttled_csd_list);

		if (cfs_rq_throttled(cfs_rq))
			unthrottle_cfs_rq(cfs_rq);

		rq_unlock_irqrestore(rq, &rf);
	}
	WARN_ON_ONCE(!list_empty(&local_unthrottle));

	rcu_read_unlock();

	return throttled;
}

/*
 * Responsible for refilling a task_group's bandwidth and unthrottling its
 * cfs_rqs as appropriate. If there has been no activity within the last
 * period the timer is deactivated until scheduling resumes; cfs_b->idle is
 * used to track this state.
 */
/*
 * do_sched_cfs_period_timer - 周期到期时补充配额并恢复受限队列；
 * 无需继续计时时返回 1。
 */
static int do_sched_cfs_period_timer(struct cfs_bandwidth *cfs_b, int overrun, unsigned long flags)
{
	int throttled;

	/* no need to continue the timer with no bandwidth constraint */
	if (cfs_b->quota == RUNTIME_INF)
		goto out_deactivate;

	throttled = !list_empty(&cfs_b->throttled_cfs_rq);
	cfs_b->nr_periods += overrun;

	/* Refill extra burst quota even if cfs_b->idle */
	__refill_cfs_bandwidth_runtime(cfs_b);

	/*
	 * idle depends on !throttled (for the case of a large deficit), and if
	 * we're going inactive then everything else can be deferred
	 */
	if (cfs_b->idle && !throttled)
		goto out_deactivate;

	if (!throttled) {
		/* mark as potentially idle for the upcoming period */
		cfs_b->idle = 1;
		return 0;
	}

	/* account preceding periods in which throttling occurred */
	cfs_b->nr_throttled += overrun;

	/*
	 * This check is repeated as we release cfs_b->lock while we unthrottle.
	 */
	while (throttled && cfs_b->runtime > 0) {
		/*
		 * 分发时需要取得各 CPU 的 rq 锁，
		 * 先释放带宽锁以遵循 rq 到带宽锁的获取顺序。
		 */
		raw_spin_unlock_irqrestore(&cfs_b->lock, flags);
		/* we can't nest cfs_b->lock while distributing bandwidth */
		throttled = distribute_cfs_runtime(cfs_b);
		raw_spin_lock_irqsave(&cfs_b->lock, flags);
	}

	/*
	 * While we are ensured activity in the period following an
	 * unthrottle, this also covers the case in which the new bandwidth is
	 * insufficient to cover the existing bandwidth deficit.  (Forcing the
	 * timer to remain active while there are any throttled entities.)
	 */
	cfs_b->idle = 0;

	return 0;

out_deactivate:
	return 1;
}

/* a cfs_rq won't donate quota below this amount */
static const u64 min_cfs_rq_runtime = 1 * NSEC_PER_MSEC;
/* minimum remaining period time to redistribute slack quota */
static const u64 min_bandwidth_expiration = 2 * NSEC_PER_MSEC;
/* how long we wait to gather additional slack before distributing */
static const u64 cfs_bandwidth_slack_period = 5 * NSEC_PER_MSEC;

/*
 * Are we near the end of the current quota period?
 *
 * Requires cfs_b->lock for hrtimer_expires_remaining to be safe against the
 * hrtimer base being cleared by hrtimer_start. In the case of
 * migrate_hrtimers, base is never cleared, so we are fine.
 */
/*
 * runtime_refresh_within - 判断周期配额刷新是否正在进行，
 * 或将在指定时间内发生。
 */
static int runtime_refresh_within(struct cfs_bandwidth *cfs_b, u64 min_expire)
{
	struct hrtimer *refresh_timer = &cfs_b->period_timer;
	s64 remaining;

	/* if the call-back is running a quota refresh is already occurring */
	if (hrtimer_callback_running(refresh_timer))
		return 1;

	/* is a quota refresh about to occur? */
	remaining = ktime_to_ns(hrtimer_expires_remaining(refresh_timer));
	if (remaining < (s64)min_expire)
		return 1;

	return 0;
}

/*
 * start_cfs_slack_bandwidth - 安排短暂延迟后重新分配闲置配额，
 * 避开即将发生的周期刷新。
 */
static void start_cfs_slack_bandwidth(struct cfs_bandwidth *cfs_b)
{
	u64 min_left = cfs_bandwidth_slack_period + min_bandwidth_expiration;

	/* if there's a quota refresh soon don't bother with slack */
	if (runtime_refresh_within(cfs_b, min_left))
		return;

	/* don't push forwards an existing deferred unthrottle */
	if (cfs_b->slack_started)
		return;
	cfs_b->slack_started = true;

	hrtimer_start(&cfs_b->slack_timer,
			ns_to_ktime(cfs_bandwidth_slack_period),
			HRTIMER_MODE_REL);
}

/* we know any runtime found here is valid as update_curr() precedes return */
/*
 * __return_cfs_rq_runtime - 为本地保留少量运行时间，
 * 将多余余额归还组配额池。
 */
static void __return_cfs_rq_runtime(struct cfs_rq *cfs_rq)
{
	struct cfs_bandwidth *cfs_b = tg_cfs_bandwidth(cfs_rq->tg);
	s64 slack_runtime = cfs_rq->runtime_remaining - min_cfs_rq_runtime;

	if (slack_runtime <= 0)
		return;

	raw_spin_lock(&cfs_b->lock);
	if (cfs_b->quota != RUNTIME_INF) {
		cfs_b->runtime += slack_runtime;

		/* we are under rq->lock, defer unthrottling using a timer */
		if (cfs_b->runtime > sched_cfs_bandwidth_slice() &&
		    !list_empty(&cfs_b->throttled_cfs_rq))
			start_cfs_slack_bandwidth(cfs_b);
	}
	raw_spin_unlock(&cfs_b->lock);

	/* even if it's not valid for return we don't want to try again */
	cfs_rq->runtime_remaining -= slack_runtime;
}

/*
 * return_cfs_rq_runtime - 仅在受配额约束的队列清空时归还多余运行时间。
 */
static __always_inline void return_cfs_rq_runtime(struct cfs_rq *cfs_rq)
{
	if (!cfs_bandwidth_used())
		return;

	if (!cfs_rq->runtime_enabled || cfs_rq->nr_queued)
		return;

	__return_cfs_rq_runtime(cfs_rq);
}

/*
 * This is done with a timer (instead of inline with bandwidth return) since
 * it's necessary to juggle rq->locks to unthrottle their respective cfs_rqs.
 */
/*
 * do_sched_cfs_slack_timer - 若离周期刷新尚远，
 * 将积累的闲置配额分给受限队列。
 */
static void do_sched_cfs_slack_timer(struct cfs_bandwidth *cfs_b)
{
	u64 runtime = 0, slice = sched_cfs_bandwidth_slice();
	unsigned long flags;

	/* confirm we're still not at a refresh boundary */
	raw_spin_lock_irqsave(&cfs_b->lock, flags);
	cfs_b->slack_started = false;

	if (runtime_refresh_within(cfs_b, min_bandwidth_expiration)) {
		raw_spin_unlock_irqrestore(&cfs_b->lock, flags);
		return;
	}

	if (cfs_b->quota != RUNTIME_INF && cfs_b->runtime > slice)
		runtime = cfs_b->runtime;

	raw_spin_unlock_irqrestore(&cfs_b->lock, flags);

	if (!runtime)
		return;

	distribute_cfs_runtime(cfs_b);
}

/*
 * When a group wakes up we want to make sure that its quota is not already
 * expired/exceeded, otherwise it may be allowed to steal additional ticks of
 * runtime as update_curr() throttling can not trigger until it's on-rq.
 */
/*
 * check_enqueue_throttle - 空队列重新有实体入队时检查配额，防止无预算运行。
 */
static void check_enqueue_throttle(struct cfs_rq *cfs_rq)
{
	if (!cfs_bandwidth_used())
		return;

	/* an active group must be handled by the update_curr()->put() path */
	if (!cfs_rq->runtime_enabled || cfs_rq->curr)
		return;

	/* ensure the group is not already throttled */
	if (cfs_rq_throttled(cfs_rq))
		return;

	/* update runtime allocation */
	account_cfs_rq_runtime(cfs_rq, 0);
	if (cfs_rq->runtime_remaining <= 0)
		throttle_cfs_rq(cfs_rq);
}

/*
 * sync_throttle - 创建或上线组队列时继承父队列当前的层级限流状态。
 */
static void sync_throttle(struct task_group *tg, int cpu)
{
	struct cfs_rq *pcfs_rq, *cfs_rq;

	if (!cfs_bandwidth_used())
		return;

	if (!tg->parent)
		return;

	cfs_rq = tg->cfs_rq[cpu];
	pcfs_rq = tg->parent->cfs_rq[cpu];

	cfs_rq->throttle_count = pcfs_rq->throttle_count;
	cfs_rq->throttled_clock_pelt = rq_clock_pelt(cpu_rq(cpu));

	/*
	 * It is not enough to sync the "pelt_clock_throttled" indicator
	 * with the parent cfs_rq when the hierarchy is not queued.
	 * Always join a throttled hierarchy with PELT clock throttled
	 * and leaf it to the first enqueue, or distribution to
	 * unthrottle the PELT clock.
	 */
	if (cfs_rq->throttle_count)
		cfs_rq->pelt_clock_throttled = 1;
}

/* conditionally throttle active cfs_rq's from put_prev_entity() */
/*
 * check_cfs_rq_runtime - 检查本队列运行时间是否耗尽，并按需进入限流。
 */
static bool check_cfs_rq_runtime(struct cfs_rq *cfs_rq)
{
	if (!cfs_bandwidth_used())
		return false;

	if (likely(!cfs_rq->runtime_enabled || cfs_rq->runtime_remaining > 0))
		return false;

	/*
	 * it's possible for a throttled entity to be forced into a running
	 * state (e.g. set_curr_task), in this case we're finished.
	 */
	if (cfs_rq_throttled(cfs_rq))
		return true;

	return throttle_cfs_rq(cfs_rq);
}

/*
 * sched_cfs_slack_timer - 处理一次闲置配额再分配，
 * 完成后不自动重启此定时器。
 */
static enum hrtimer_restart sched_cfs_slack_timer(struct hrtimer *timer)
{
	struct cfs_bandwidth *cfs_b =
		container_of(timer, struct cfs_bandwidth, slack_timer);

	do_sched_cfs_slack_timer(cfs_b);

	return HRTIMER_NORESTART;
}

/*
 * sched_cfs_period_timer - 推进周期定时器并补充预算；空闲时停表，
 * 持续超时时放大周期和配额。
 */
static enum hrtimer_restart sched_cfs_period_timer(struct hrtimer *timer)
{
	struct cfs_bandwidth *cfs_b =
		container_of(timer, struct cfs_bandwidth, period_timer);
	unsigned long flags;
	int overrun;
	int idle = 0;
	int count = 0;

	raw_spin_lock_irqsave(&cfs_b->lock, flags);
	for (;;) {
		overrun = hrtimer_forward_now(timer, cfs_b->period);
		if (!overrun)
			break;

		idle = do_sched_cfs_period_timer(cfs_b, overrun, flags);

		if (++count > 3) {
			u64 new, old = ktime_to_ns(cfs_b->period);

			/*
			 * Grow period by a factor of 2 to avoid losing precision.
			 * Precision loss in the quota/period ratio can cause __cfs_schedulable
			 * to fail.
			 */
			new = old * 2;
			if (new < max_bw_quota_period_us * NSEC_PER_USEC) {
				cfs_b->period = ns_to_ktime(new);
				cfs_b->quota *= 2;
				cfs_b->burst *= 2;

				pr_warn_ratelimited(
	"cfs_period_timer[cpu%d]: period too short, scaling up (new cfs_period_us = %lld, cfs_quota_us = %lld)\n",
					smp_processor_id(),
					div_u64(new, NSEC_PER_USEC),
					div_u64(cfs_b->quota, NSEC_PER_USEC));
			} else {
				pr_warn_ratelimited(
	"cfs_period_timer[cpu%d]: period too short, but cannot scale up without losing precision (cfs_period_us = %lld, cfs_quota_us = %lld)\n",
					smp_processor_id(),
					div_u64(old, NSEC_PER_USEC),
					div_u64(cfs_b->quota, NSEC_PER_USEC));
			}

			/* reset count so we don't come right back in here */
			count = 0;
		}
	}
	if (idle)
		cfs_b->period_active = 0;
	raw_spin_unlock_irqrestore(&cfs_b->lock, flags);

	return idle ? HRTIMER_NORESTART : HRTIMER_RESTART;
}

/*
 * init_cfs_bandwidth：
 * 初始化任务组的无限额默认状态、层级配额及两类带宽定时器。
 */
void init_cfs_bandwidth(struct cfs_bandwidth *cfs_b, struct cfs_bandwidth *parent)
{
	raw_spin_lock_init(&cfs_b->lock);
	cfs_b->runtime = 0;
	cfs_b->quota = RUNTIME_INF;
	cfs_b->period = us_to_ktime(default_bw_period_us());
	cfs_b->burst = 0;
	cfs_b->hierarchical_quota = parent ? parent->hierarchical_quota : RUNTIME_INF;

	INIT_LIST_HEAD(&cfs_b->throttled_cfs_rq);
	hrtimer_setup(&cfs_b->period_timer, sched_cfs_period_timer, CLOCK_MONOTONIC,
		      HRTIMER_MODE_ABS_PINNED);

	/* Add a random offset so that timers interleave */
	hrtimer_set_expires(&cfs_b->period_timer,
			    get_random_u32_below(cfs_b->period));
	hrtimer_setup(&cfs_b->slack_timer, sched_cfs_slack_timer, CLOCK_MONOTONIC,
		      HRTIMER_MODE_REL);
	cfs_b->slack_started = false;
}

/*
 * init_cfs_rq_runtime - 初始化本地队列的带宽开关、限流队列和等待任务链表。
 */
static void init_cfs_rq_runtime(struct cfs_rq *cfs_rq)
{
	cfs_rq->runtime_enabled = 0;
	INIT_LIST_HEAD(&cfs_rq->throttled_list);
	INIT_LIST_HEAD(&cfs_rq->throttled_csd_list);
	INIT_LIST_HEAD(&cfs_rq->throttled_limbo_list);
}

/*
 * start_cfs_bandwidth - 在带宽锁保护下启动尚未运行的周期配额定时器。
 */
void start_cfs_bandwidth(struct cfs_bandwidth *cfs_b)
{
	lockdep_assert_held(&cfs_b->lock);

	if (cfs_b->period_active)
		return;

	cfs_b->period_active = 1;
	hrtimer_forward_now(&cfs_b->period_timer, cfs_b->period);
	hrtimer_start_expires(&cfs_b->period_timer, HRTIMER_MODE_ABS_PINNED);
}

/*
 * destroy_cfs_bandwidth - 取消带宽定时器，
 * 并处理残留的异步解限请求以便回收对象。
 */
static void destroy_cfs_bandwidth(struct cfs_bandwidth *cfs_b)
{
	int __maybe_unused i;

	/* init_cfs_bandwidth() was not called */
	if (!cfs_b->throttled_cfs_rq.next)
		return;

	hrtimer_cancel(&cfs_b->period_timer);
	hrtimer_cancel(&cfs_b->slack_timer);

	/*
	 * It is possible that we still have some cfs_rq's pending on a CSD
	 * list, though this race is very rare. In order for this to occur, we
	 * must have raced with the last task leaving the group while there
	 * exist throttled cfs_rq(s), and the period_timer must have queued the
	 * CSD item but the remote cpu has not yet processed it. To handle this,
	 * we can simply flush all pending CSD work inline here. We're
	 * guaranteed at this point that no additional cfs_rq of this group can
	 * join a CSD list.
	 */
	for_each_possible_cpu(i) {
		struct rq *rq = cpu_rq(i);
		unsigned long flags;

		if (list_empty(&rq->cfsb_csd_list))
			continue;

		local_irq_save(flags);
		__cfsb_csd_unthrottle(rq);
		local_irq_restore(flags);
	}
}

/*
 * Both these CPU hotplug callbacks race against unregister_fair_sched_group()
 *
 * The race is harmless, since modifying bandwidth settings of unhooked group
 * bits doesn't do much.
 */

/* cpu online callback */
/*
 * update_runtime_enabled - CPU 上线时，
 * 按各任务组是否配置有限配额刷新本地控制开关。
 */
static void __maybe_unused update_runtime_enabled(struct rq *rq)
{
	struct task_group *tg;

	lockdep_assert_rq_held(rq);

	rcu_read_lock();
	list_for_each_entry_rcu(tg, &task_groups, list) {
		struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;
		struct cfs_rq *cfs_rq = tg->cfs_rq[cpu_of(rq)];

		raw_spin_lock(&cfs_b->lock);
		cfs_rq->runtime_enabled = cfs_b->quota != RUNTIME_INF;
		raw_spin_unlock(&cfs_b->lock);
	}
	rcu_read_unlock();
}

/* cpu offline callback */
/*
 * unthrottle_offline_cfs_rqs - CPU 离线时关闭本地带宽限制并恢复受限任务，
 * 使后续迁出可继续。
 */
static void __maybe_unused unthrottle_offline_cfs_rqs(struct rq *rq)
{
	struct task_group *tg;

	lockdep_assert_rq_held(rq);

	// Do not unthrottle for an active CPU
	if (cpumask_test_cpu(cpu_of(rq), cpu_active_mask))
		return;

	/*
	 * The rq clock has already been updated in the
	 * set_rq_offline(), so we should skip updating
	 * the rq clock again in unthrottle_cfs_rq().
	 */
	rq_clock_start_loop_update(rq);

	rcu_read_lock();
	list_for_each_entry_rcu(tg, &task_groups, list) {
		struct cfs_rq *cfs_rq = tg->cfs_rq[cpu_of(rq)];

		if (!cfs_rq->runtime_enabled)
			continue;

		/*
		 * Offline rq is schedulable till CPU is completely disabled
		 * in take_cpu_down(), so we prevent new cfs throttling here.
		 */
		cfs_rq->runtime_enabled = 0;

		if (!cfs_rq_throttled(cfs_rq))
			continue;

		/*
		 * clock_task is not advancing so we just need to make sure
		 * there's some valid quota amount
		 */
		cfs_rq->runtime_remaining = 1;
		unthrottle_cfs_rq(cfs_rq);
	}
	rcu_read_unlock();

	rq_clock_stop_loop_update(rq);
}

/*
 * cfs_task_bw_constrained - 检查任务是否受本组或祖先组的带宽配额约束。
 */
bool cfs_task_bw_constrained(struct task_struct *p)
{
	struct cfs_rq *cfs_rq = task_cfs_rq(p);

	if (!cfs_bandwidth_used())
		return false;

	if (cfs_rq->runtime_enabled ||
	    tg_cfs_bandwidth(cfs_rq->tg)->hierarchical_quota != RUNTIME_INF)
		return true;

	return false;
}

#ifdef CONFIG_NO_HZ_FULL
/* called from pick_next_task_fair() */
/*
 * sched_fair_update_stop_tick - 全动态 tick CPU 只有一个任务时，
 * 若仍有带宽约束则保留调度 tick。
 */
static void sched_fair_update_stop_tick(struct rq *rq, struct task_struct *p)
{
	int cpu = cpu_of(rq);

	if (!cfs_bandwidth_used())
		return;

	if (!tick_nohz_full_cpu(cpu))
		return;

	if (rq->nr_running != 1)
		return;

	/*
	 *  We know there is only one task runnable and we've just picked it. The
	 *  normal enqueue path will have cleared TICK_DEP_BIT_SCHED if we will
	 *  be otherwise able to stop the tick. Just need to check if we are using
	 *  bandwidth control.
	 */
	if (cfs_task_bw_constrained(p))
		tick_nohz_dep_set_cpu(cpu, TICK_DEP_BIT_SCHED);
}
#endif /* CONFIG_NO_HZ_FULL */

#else /* !CONFIG_CFS_BANDWIDTH: */

/*
 * account_cfs_rq_runtime - 未启用 CFS 带宽控制时，无需扣减配额。
 */
static void account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec) {}
/*
 * check_cfs_rq_runtime - 未启用 CFS 带宽控制时，队列不会因配额不足而限流。
 */
static bool check_cfs_rq_runtime(struct cfs_rq *cfs_rq) { return false; }
/*
 * check_enqueue_throttle - 未启用 CFS 带宽控制时，无需执行入队配额检查。
 */
static void check_enqueue_throttle(struct cfs_rq *cfs_rq) {}
/*
 * sync_throttle - 未启用 CFS 带宽控制时，无父组限流状态需要继承。
 */
static inline void sync_throttle(struct task_group *tg, int cpu) {}
/*
 * return_cfs_rq_runtime - 未启用 CFS 带宽控制时，无配额需要归还。
 */
static __always_inline void return_cfs_rq_runtime(struct cfs_rq *cfs_rq) {}
/*
 * task_throttle_setup_work - 未启用 CFS 带宽控制时，不安排任务限流回调。
 */
static void task_throttle_setup_work(struct task_struct *p) {}
/*
 * task_is_throttled - 未启用 CFS 带宽控制时，任务不会处于配额限流状态。
 */
static bool task_is_throttled(struct task_struct *p) { return false; }
/*
 * dequeue_throttled_task - 未启用 CFS 带宽控制时，无限流等待链表需要维护。
 */
static void dequeue_throttled_task(struct task_struct *p, int flags) {}
/*
 * enqueue_throttled_task - 未启用 CFS 带宽控制时，入队不会被配额限流截留。
 */
static bool enqueue_throttled_task(struct task_struct *p) { return false; }
/*
 * record_throttle_clock - 未启用 CFS 带宽控制时，无需记录限流时间。
 */
static void record_throttle_clock(struct cfs_rq *cfs_rq) {}

/*
 * cfs_rq_throttled - 未启用 CFS 带宽控制时，队列限流状态恒为假。
 */
static inline int cfs_rq_throttled(struct cfs_rq *cfs_rq)
{
	return 0;
}

/*
 * cfs_rq_pelt_clock_throttled - 未启用 CFS 带宽控制时，
 * PELT 时钟不会因配额限流暂停。
 */
static inline bool cfs_rq_pelt_clock_throttled(struct cfs_rq *cfs_rq)
{
	return false;
}

/*
 * throttled_hierarchy - 未启用 CFS 带宽控制时，层级中不存在配额限流。
 */
static inline int throttled_hierarchy(struct cfs_rq *cfs_rq)
{
	return 0;
}

/*
 * lb_throttled_hierarchy - 未启用 CFS 带宽控制时，
 * 目标 CPU 不会因组配额阻止迁入。
 */
static inline int lb_throttled_hierarchy(struct task_struct *p, int dst_cpu)
{
	return 0;
}

#ifdef CONFIG_FAIR_GROUP_SCHED
/*
 * init_cfs_bandwidth - 未启用 CFS 带宽控制时，无共享配额池需要初始化。
 */
void init_cfs_bandwidth(struct cfs_bandwidth *cfs_b, struct cfs_bandwidth *parent) {}
/*
 * init_cfs_rq_runtime - 未启用 CFS 带宽控制时，无本地配额状态需要初始化。
 */
static void init_cfs_rq_runtime(struct cfs_rq *cfs_rq) {}
#endif

/*
 * tg_cfs_bandwidth - 未启用 CFS 带宽控制时，不存在组带宽对象。
 */
static inline struct cfs_bandwidth *tg_cfs_bandwidth(struct task_group *tg)
{
	return NULL;
}
/*
 * destroy_cfs_bandwidth - 未启用 CFS 带宽控制时，无带宽资源需要释放。
 */
static inline void destroy_cfs_bandwidth(struct cfs_bandwidth *cfs_b) {}
/*
 * update_runtime_enabled - 未启用 CFS 带宽控制时，无需更新本地配额开关。
 */
static inline void update_runtime_enabled(struct rq *rq) {}
/*
 * unthrottle_offline_cfs_rqs - 未启用 CFS 带宽控制时，
 * CPU 离线无需解除配额限流。
 */
static inline void unthrottle_offline_cfs_rqs(struct rq *rq) {}
#ifdef CONFIG_CGROUP_SCHED
/*
 * cfs_task_bw_constrained - 未启用 CFS 带宽控制时，任务不受此类配额约束。
 */
bool cfs_task_bw_constrained(struct task_struct *p)
{
	return false;
}
#endif
#endif /* !CONFIG_CFS_BANDWIDTH */

#if !defined(CONFIG_CFS_BANDWIDTH) || !defined(CONFIG_NO_HZ_FULL)
/*
 * sched_fair_update_stop_tick - 未启用 CFS 带宽控制时，
 * 不因配额记账而增加 tick 依赖。
 */
static inline void sched_fair_update_stop_tick(struct rq *rq, struct task_struct *p) {}
#endif

/**************************************************
 * CFS operations on tasks:
 */

#ifdef CONFIG_SCHED_HRTICK
/*
 * hrtick_start_fair - 根据实体本轮剩余时间片设置高精度 tick，
 * 已超时时请求重调度。
 */
static void hrtick_start_fair(struct rq *rq, struct task_struct *p)
{
	struct sched_entity *se = &p->se;

	WARN_ON_ONCE(task_rq(p) != rq);

	if (rq->cfs.h_nr_queued > 1) {
		u64 ran = se->sum_exec_runtime - se->prev_sum_exec_runtime;
		u64 slice = se->slice;
		s64 delta = slice - ran;

		if (delta < 0) {
			if (task_current_donor(rq, p))
				resched_curr(rq);
			return;
		}
		hrtick_start(rq, delta);
	}
}

/*
 * called from enqueue/dequeue and updates the hrtick when the
 * current task is from our class and nr_running is low enough
 * to matter.
 */
/*
 * hrtick_update：
 * 仅在高精度 tick 可用且资源提供者属于公平类时重新设置到期点。
 */
static void hrtick_update(struct rq *rq)
{
	struct task_struct *donor = rq->donor;

	if (!hrtick_enabled_fair(rq) || donor->sched_class != &fair_sched_class)
		return;

	hrtick_start_fair(rq, donor);
}
#else /* !CONFIG_SCHED_HRTICK: */
/*
 * hrtick_start_fair - 未启用高精度调度 tick 时，
 * 不单独安排时间片到期定时器。
 */
static inline void
hrtick_start_fair(struct rq *rq, struct task_struct *p)
{
}

/*
 * hrtick_update - 未启用高精度调度 tick 时，无到期点需要更新。
 */
static inline void hrtick_update(struct rq *rq)
{
}
#endif /* !CONFIG_SCHED_HRTICK */

/*
 * cpu_overutilized - 在能耗调度启用时，
 * 结合 uclamp 判断 CPU 利用率是否超出容量。
 */
static inline bool cpu_overutilized(int cpu)
{
	unsigned long  rq_util_min, rq_util_max;

	if (!sched_energy_enabled())
		return false;

	rq_util_min = uclamp_rq_get(cpu_rq(cpu), UCLAMP_MIN);
	rq_util_max = uclamp_rq_get(cpu_rq(cpu), UCLAMP_MAX);

	/* Return true only if the utilization doesn't fit CPU's capacity */
	return !util_fits_cpu(cpu_util_cfs(cpu), rq_util_min, rq_util_max, cpu);
}

/*
 * overutilized value make sense only if EAS is enabled
 */
/*
 * is_rd_overutilized - 读取根调度域的过度利用标记；
 * 未启用能耗调度时走常规均衡路径。
 */
static inline bool is_rd_overutilized(struct root_domain *rd)
{
	return !sched_energy_enabled() || READ_ONCE(rd->overutilized);
}

/*
 * set_rd_overutilized - 更新能耗调度使用的根域过度利用状态并发出跟踪事件。
 */
static inline void set_rd_overutilized(struct root_domain *rd, bool flag)
{
	if (!sched_energy_enabled())
		return;

	WRITE_ONCE(rd->overutilized, flag);
	trace_sched_overutilized_tp(rd, flag);
}

/*
 * check_update_overutilized_status - 任务入队后若本 CPU 超出容量，
 * 将根域标记为过度利用。
 */
static inline void check_update_overutilized_status(struct rq *rq)
{
	/*
	 * overutilized field is used for load balancing decisions only
	 * if energy aware scheduler is being used
	 */

	if (!is_rd_overutilized(rq->rd) && cpu_overutilized(rq->cpu))
		set_rd_overutilized(rq->rd, 1);
}

/* Runqueue only has SCHED_IDLE tasks enqueued */
/*
 * sched_idle_rq - 判断非空运行队列是否仅有空闲策略任务，
 * 区别于完全没有任务。
 */
static int sched_idle_rq(struct rq *rq)
{
	return unlikely(rq->nr_running == rq->cfs.h_nr_idle &&
			rq->nr_running);
}

/*
 * sched_idle_cpu - 按运行队列统计判断指定 CPU 是否只承担空闲策略任务。
 */
static int sched_idle_cpu(int cpu)
{
	return sched_idle_rq(cpu_rq(cpu));
}

/*
 * requeue_delayed_entity - 唤醒尚未真正出队的实体，
 * 撤销延迟标记并按需将正 lag 归零。
 */
static void
requeue_delayed_entity(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);

	/*
	 * se->sched_delayed should imply: se->on_rq == 1.
	 * Because a delayed entity is one that is still on
	 * the runqueue competing until elegibility.
	 */
	WARN_ON_ONCE(!se->sched_delayed);
	WARN_ON_ONCE(!se->on_rq);

	if (sched_feat(DELAY_ZERO)) {
		update_entity_lag(cfs_rq, se);
		if (se->vlag > 0) {
			cfs_rq->nr_queued--;
			if (se != cfs_rq->curr)
				__dequeue_entity(cfs_rq, se);
			se->vlag = 0;
			place_entity(cfs_rq, se, 0);
			if (se != cfs_rq->curr)
				__enqueue_entity(cfs_rq, se);
			cfs_rq->nr_queued++;
		}
	}

	update_load_avg(cfs_rq, se, 0);
	clear_delayed(se);
}

/*
 * The enqueue_task method is called before nr_running is
 * increased. Here we update the fair scheduling stats and
 * then put the task into the rbtree:
 */
/*
 * enqueue_task_fair - 公平类入队入口，
 * 沿任务到根的层级维护实体、任务数和最短时间片。
 * 同时处理限流等待、延迟唤醒、利用率预测和 fair_server 启动。
 */
static void
enqueue_task_fair(struct rq *rq, struct task_struct *p, int flags)
{
	struct cfs_rq *cfs_rq;
	struct sched_entity *se = &p->se;
	int h_nr_idle = task_has_idle_policy(p);
	int h_nr_runnable = 1;
	int task_new = !(flags & ENQUEUE_WAKEUP);
	int rq_h_nr_queued = rq->cfs.h_nr_queued;
	u64 slice = 0;

	if (task_is_throttled(p) && enqueue_throttled_task(p))
		return;

	/*
	 * The code below (indirectly) updates schedutil which looks at
	 * the cfs_rq utilization to select a frequency.
	 * Let's add the task's estimated utilization to the cfs_rq's
	 * estimated utilization, before we update schedutil.
	 */
	if (!p->se.sched_delayed || (flags & ENQUEUE_DELAYED))
		util_est_enqueue(&rq->cfs, p);

	/*
	 * 实体还留在原队列时，唤醒仅恢复延迟状态，
	 * 不能重复增加队列权重和排队数。
	 */
	if (flags & ENQUEUE_DELAYED) {
		requeue_delayed_entity(se);
		return;
	}

	/*
	 * If in_iowait is set, the code below may not trigger any cpufreq
	 * utilization updates, so do it here explicitly with the IOWAIT flag
	 * passed.
	 */
	if (p->in_iowait)
		cpufreq_update_util(rq, SCHED_CPUFREQ_IOWAIT);

	if (task_new && se->sched_delayed)
		h_nr_runnable = 0;

	/*
	 * 先把尚未在队的实体逐层接入；遇到已在队的祖先后，
	 * 改为只更新统计和时间片。
	 */
	for_each_sched_entity(se) {
		if (se->on_rq) {
			if (se->sched_delayed)
				requeue_delayed_entity(se);
			break;
		}
		cfs_rq = cfs_rq_of(se);

		/*
		 * Basically set the slice of group entries to the min_slice of
		 * their respective cfs_rq. This ensures the group can service
		 * its entities in the desired time-frame.
		 */
		if (slice) {
			se->slice = slice;
			se->custom_slice = 1;
		}
		enqueue_entity(cfs_rq, se, flags);
		slice = cfs_rq_min_slice(cfs_rq);

		cfs_rq->h_nr_runnable += h_nr_runnable;
		cfs_rq->h_nr_queued++;
		cfs_rq->h_nr_idle += h_nr_idle;

		if (cfs_rq_is_idle(cfs_rq))
			h_nr_idle = 1;

		flags = ENQUEUE_WAKEUP;
	}

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);

		update_load_avg(cfs_rq, se, UPDATE_TG);
		se_update_runnable(se);
		update_cfs_group(se);

		se->slice = slice;
		if (se != cfs_rq->curr)
			min_vruntime_cb_propagate(&se->run_node, NULL);
		slice = cfs_rq_min_slice(cfs_rq);

		cfs_rq->h_nr_runnable += h_nr_runnable;
		cfs_rq->h_nr_queued++;
		cfs_rq->h_nr_idle += h_nr_idle;

		if (cfs_rq_is_idle(cfs_rq))
			h_nr_idle = 1;
	}

	if (!rq_h_nr_queued && rq->cfs.h_nr_queued) {
		/* Account for idle runtime */
		if (!rq->nr_running)
			dl_server_update_idle_time(rq, rq->curr);
		dl_server_start(&rq->fair_server);
	}

	/* At this point se is NULL and we are at root level*/
	add_nr_running(rq, 1);

	/*
	 * Since new tasks are assigned an initial util_avg equal to
	 * half of the spare capacity of their CPU, tiny tasks have the
	 * ability to cross the overutilized threshold, which will
	 * result in the load balancer ruining all the task placement
	 * done by EAS. As a way to mitigate that effect, do not account
	 * for the first enqueue operation of new tasks during the
	 * overutilized flag detection.
	 *
	 * A better way of solving this problem would be to wait for
	 * the PELT signals of tasks to converge before taking them
	 * into account, but that is not straightforward to implement,
	 * and the following generally works well enough in practice.
	 */
	if (!task_new)
		check_update_overutilized_status(rq);

	assert_list_leaf_cfs_rq(rq);

	hrtick_update(rq);
}

/*
 * Basically dequeue_task_fair(), except it can deal with dequeue_entity()
 * failing half-way through and resume the dequeue later.
 *
 * Returns:
 * -1 - dequeue delayed
 *  0 - dequeue throttled
 *  1 - dequeue complete
 */
/*
 * dequeue_entities - 沿组层级移除任务及变空的组实体，
 * 并更新其余祖先的统计。
 * 任务本身被延迟出队时返回 -1；完成延迟出队时还需调用 __block_task()。
 */
static int dequeue_entities(struct rq *rq, struct sched_entity *se, int flags)
{
	bool was_sched_idle = sched_idle_rq(rq);
	bool task_sleep = flags & DEQUEUE_SLEEP;
	bool task_delayed = flags & DEQUEUE_DELAYED;
	bool task_throttled = flags & DEQUEUE_THROTTLE;
	struct task_struct *p = NULL;
	int h_nr_idle = 0;
	int h_nr_queued = 0;
	int h_nr_runnable = 0;
	struct cfs_rq *cfs_rq;
	u64 slice = 0;

	if (entity_is_task(se)) {
		p = task_of(se);
		h_nr_queued = 1;
		h_nr_idle = task_has_idle_policy(p);
		if (task_sleep || task_delayed || !se->sched_delayed)
			h_nr_runnable = 1;
	}

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);

		if (!dequeue_entity(cfs_rq, se, flags)) {
			if (p && &p->se == se)
				return -1;

			slice = cfs_rq_min_slice(cfs_rq);
			break;
		}

		cfs_rq->h_nr_runnable -= h_nr_runnable;
		cfs_rq->h_nr_queued -= h_nr_queued;
		cfs_rq->h_nr_idle -= h_nr_idle;

		if (cfs_rq_is_idle(cfs_rq))
			h_nr_idle = h_nr_queued;

		if (throttled_hierarchy(cfs_rq) && task_throttled)
			record_throttle_clock(cfs_rq);

		/* Don't dequeue parent if it has other entities besides us */
		if (cfs_rq->load.weight) {
			slice = cfs_rq_min_slice(cfs_rq);

			/* Avoid re-evaluating load for this entity: */
			se = parent_entity(se);
			/*
			 * Bias pick_next to pick a task from this cfs_rq, as
			 * p is sleeping when it is within its sched_slice.
			 */
			if (task_sleep && se)
				set_next_buddy(se);
			break;
		}
		flags |= DEQUEUE_SLEEP;
		flags &= ~(DEQUEUE_DELAYED | DEQUEUE_SPECIAL);
	}

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);

		update_load_avg(cfs_rq, se, UPDATE_TG);
		se_update_runnable(se);
		update_cfs_group(se);

		se->slice = slice;
		if (se != cfs_rq->curr)
			min_vruntime_cb_propagate(&se->run_node, NULL);
		slice = cfs_rq_min_slice(cfs_rq);

		cfs_rq->h_nr_runnable -= h_nr_runnable;
		cfs_rq->h_nr_queued -= h_nr_queued;
		cfs_rq->h_nr_idle -= h_nr_idle;

		if (cfs_rq_is_idle(cfs_rq))
			h_nr_idle = h_nr_queued;

		if (throttled_hierarchy(cfs_rq) && task_throttled)
			record_throttle_clock(cfs_rq);
	}

	sub_nr_running(rq, h_nr_queued);

	/* balance early to pull high priority tasks */
	if (unlikely(!was_sched_idle && sched_idle_rq(rq)))
		rq->next_balance = jiffies;

	if (p && task_delayed) {
		WARN_ON_ONCE(!task_sleep);
		WARN_ON_ONCE(p->on_rq != 1);

		/* Fix-up what dequeue_task_fair() skipped */
		hrtick_update(rq);

		/*
		 * Fix-up what block_task() skipped.
		 *
		 * Must be last, @p might not be valid after this.
		 */
		/*
		 * 延迟到此刻才完成调度核心的阻塞收尾，
		 * 之后不可再解引用可能被释放的任务。
		 */
		__block_task(rq, p);
	}

	return 1;
}

/*
 * The dequeue_task method is called before nr_running is
 * decreased. We remove the task from the rbtree and
 * update the fair scheduling stats:
 */
/*
 * dequeue_task_fair - 公平类出队入口，撤销利用率预测并驱动层级出队。
 * 返回 false 表示实体暂留队列，尚未完成真正的阻塞处理。
 */
static bool dequeue_task_fair(struct rq *rq, struct task_struct *p, int flags)
{
	if (task_is_throttled(p)) {
		dequeue_throttled_task(p, flags);
		return true;
	}

	if (!p->se.sched_delayed)
		util_est_dequeue(&rq->cfs, p);

	util_est_update(&rq->cfs, p, flags & DEQUEUE_SLEEP);
	if (dequeue_entities(rq, &p->se, flags) < 0)
		return false;

	/*
	 * Must not reference @p after dequeue_entities(DEQUEUE_DELAYED).
	 */

	hrtick_update(rq);
	return true;
}

/*
 * cfs_h_nr_delayed - 用层级排队数减去可运行数，得到延迟出队任务数。
 */
static inline unsigned int cfs_h_nr_delayed(struct rq *rq)
{
	return (rq->cfs.h_nr_queued - rq->cfs.h_nr_runnable);
}

/* Working cpumask for: sched_balance_rq(), sched_balance_newidle(). */
static DEFINE_PER_CPU(cpumask_var_t, load_balance_mask);
static DEFINE_PER_CPU(cpumask_var_t, select_rq_mask);
static DEFINE_PER_CPU(cpumask_var_t, should_we_balance_tmpmask);

#ifdef CONFIG_NO_HZ_COMMON

static struct {
	cpumask_var_t idle_cpus_mask;
	atomic_t nr_cpus;
	int has_blocked;		/* Idle CPUS has blocked load */
	int needs_update;		/* Newly idle CPUs need their next_balance collated */
	unsigned long next_balance;     /* in jiffy units */
	unsigned long next_blocked;	/* Next update of blocked load in jiffies */
} nohz ____cacheline_aligned;

#endif /* CONFIG_NO_HZ_COMMON */

/*
 * cpu_load - 读取 CPU 根公平队列的加权 PELT 负载，包含未衰减完的历史贡献。
 */
static unsigned long cpu_load(struct rq *rq)
{
	return cfs_rq_load_avg(&rq->cfs);
}

/*
 * cpu_load_without - compute CPU load without any contributions from *p
 * @cpu: the CPU which load is requested
 * @p: the task which load should be discounted
 *
 * The load of a CPU is defined by the load of tasks currently enqueued on that
 * CPU as well as tasks which are currently sleeping after an execution on that
 * CPU.
 *
 * This method returns the load of the specified CPU by discounting the load of
 * the specified task, whenever the task is currently contributing to the CPU
 * load.
 */
/*
 * cpu_load_without - 估算排除指定任务后 CPU 的负载，
 * 避免选核时重复计算任务自身。
 */
static unsigned long cpu_load_without(struct rq *rq, struct task_struct *p)
{
	struct cfs_rq *cfs_rq;
	unsigned int load;

	/* Task has no contribution or is new */
	if (cpu_of(rq) != task_cpu(p) || !READ_ONCE(p->se.avg.last_update_time))
		return cpu_load(rq);

	cfs_rq = &rq->cfs;
	load = READ_ONCE(cfs_rq->avg.load_avg);

	/* Discount task's util from CPU's util */
	lsub_positive(&load, task_h_load(p));

	return load;
}

/*
 * cpu_runnable - 读取 CPU 上公平任务的平滑可运行需求。
 */
static unsigned long cpu_runnable(struct rq *rq)
{
	return cfs_rq_runnable_avg(&rq->cfs);
}

/*
 * cpu_runnable_without - 估算扣除指定任务贡献后的可运行需求。
 */
static unsigned long cpu_runnable_without(struct rq *rq, struct task_struct *p)
{
	struct cfs_rq *cfs_rq;
	unsigned int runnable;

	/* Task has no contribution or is new */
	if (cpu_of(rq) != task_cpu(p) || !READ_ONCE(p->se.avg.last_update_time))
		return cpu_runnable(rq);

	cfs_rq = &rq->cfs;
	runnable = READ_ONCE(cfs_rq->avg.runnable_avg);

	/* Discount task's runnable from CPU's runnable */
	lsub_positive(&runnable, p->se.avg.runnable_avg);

	return runnable;
}

/*
 * capacity_of - 读取均衡器维护的 CPU 可用容量。
 */
static unsigned long capacity_of(int cpu)
{
	return cpu_rq(cpu)->cpu_capacity;
}

/*
 * record_wakee - 记录唤醒目标变化次数并定期衰减，
 * 用于识别多对多唤醒模式。
 */
static void record_wakee(struct task_struct *p)
{
	/*
	 * Only decay a single time; tasks that have less then 1 wakeup per
	 * jiffy will not have built up many flips.
	 */
	if (time_after(jiffies, current->wakee_flip_decay_ts + HZ)) {
		current->wakee_flips >>= 1;
		current->wakee_flip_decay_ts = jiffies;
	}

	if (current->last_wakee != p) {
		current->last_wakee = p;
		current->wakee_flips++;
	}
}

/*
 * Detect M:N waker/wakee relationships via a switching-frequency heuristic.
 *
 * A waker of many should wake a different task than the one last awakened
 * at a frequency roughly N times higher than one of its wakees.
 *
 * In order to determine whether we should let the load spread vs consolidating
 * to shared cache, we look for a minimum 'flip' frequency of llc_size in one
 * partner, and a factor of lls_size higher frequency in the other.
 *
 * With both conditions met, we can be relatively sure that the relationship is
 * non-monogamous, with partner count exceeding socket size.
 *
 * Waker/wakee being client/server, worker/dispatcher, interrupt source or
 * whatever is irrelevant, spread criteria is apparent partner count exceeds
 * socket size.
 */
/*
 * wake_wide - 根据双方唤醒目标变化和共享缓存规模，
 * 判断是否应分散唤醒任务。
 */
static int wake_wide(struct task_struct *p)
{
	unsigned int master = current->wakee_flips;
	unsigned int slave = p->wakee_flips;
	int factor = __this_cpu_read(sd_llc_size);

	if (master < slave)
		swap(master, slave);
	if (slave < factor || master < slave * factor)
		return 0;
	return 1;
}

/*
 * The purpose of wake_affine() is to quickly determine on which CPU we can run
 * soonest. For the purpose of speed we only consider the waking and previous
 * CPU.
 *
 * wake_affine_idle() - only considers 'now', it check if the waking CPU is
 *			cache-affine and is (or	will be) idle.
 *
 * wake_affine_weight() - considers the weight to reflect the average
 *			  scheduling latency of the CPUs. This seems to work
 *			  for the overloaded case.
 */
/*
 * wake_affine_idle - 根据空闲状态、缓存共享和同步唤醒提示，
 * 在当前与上次 CPU 间选择。
 */
static int
wake_affine_idle(int this_cpu, int prev_cpu, int sync)
{
	/*
	 * If this_cpu is idle, it implies the wakeup is from interrupt
	 * context. Only allow the move if cache is shared. Otherwise an
	 * interrupt intensive workload could force all tasks onto one
	 * node depending on the IO topology or IRQ affinity settings.
	 *
	 * If the prev_cpu is idle and cache affine then avoid a migration.
	 * There is no guarantee that the cache hot data from an interrupt
	 * is more important than cache hot data on the prev_cpu and from
	 * a cpufreq perspective, it's better to have higher utilisation
	 * on one CPU.
	 */
	if (available_idle_cpu(this_cpu) && cpus_share_cache(this_cpu, prev_cpu))
		return available_idle_cpu(prev_cpu) ? prev_cpu : this_cpu;

	if (sync) {
		struct rq *rq = cpu_rq(this_cpu);

		if ((rq->nr_running - cfs_h_nr_delayed(rq)) == 1)
			return this_cpu;
	}

	if (available_idle_cpu(prev_cpu))
		return prev_cpu;

	return nr_cpumask_bits;
}

/*
 * wake_affine_weight - 比较迁入后按容量归一化的两端负载，
 * 判断是否偏向唤醒者 CPU。
 */
static int
wake_affine_weight(struct sched_domain *sd, struct task_struct *p,
		   int this_cpu, int prev_cpu, int sync)
{
	s64 this_eff_load, prev_eff_load;
	unsigned long task_load;

	this_eff_load = cpu_load(cpu_rq(this_cpu));

	if (sync) {
		unsigned long current_load = task_h_load(current);

		if (current_load > this_eff_load)
			return this_cpu;

		this_eff_load -= current_load;
	}

	task_load = task_h_load(p);

	this_eff_load += task_load;
	if (sched_feat(WA_BIAS))
		this_eff_load *= 100;
	this_eff_load *= capacity_of(prev_cpu);

	prev_eff_load = cpu_load(cpu_rq(prev_cpu));
	prev_eff_load -= task_load;
	if (sched_feat(WA_BIAS))
		prev_eff_load *= 100 + (sd->imbalance_pct - 100) / 2;
	prev_eff_load *= capacity_of(this_cpu);

	/*
	 * If sync, adjust the weight of prev_eff_load such that if
	 * prev_eff == this_eff that select_idle_sibling() will consider
	 * stacking the wakee on top of the waker if no other CPU is
	 * idle.
	 */
	if (sync)
		prev_eff_load += 1;

	return this_eff_load < prev_eff_load ? this_cpu : nr_cpumask_bits;
}

/*
 * wake_affine - 组合空闲和负载启发式，
 * 在保持唤醒局部性与分散负载之间取舍。
 */
static int wake_affine(struct sched_domain *sd, struct task_struct *p,
		       int this_cpu, int prev_cpu, int sync)
{
	int target = nr_cpumask_bits;

	if (sched_feat(WA_IDLE))
		target = wake_affine_idle(this_cpu, prev_cpu, sync);

	if (sched_feat(WA_WEIGHT) && target == nr_cpumask_bits)
		target = wake_affine_weight(sd, p, this_cpu, prev_cpu, sync);

	schedstat_inc(p->stats.nr_wakeups_affine_attempts);
	if (target != this_cpu)
		return prev_cpu;

	schedstat_inc(sd->ttwu_move_affine);
	schedstat_inc(p->stats.nr_wakeups_affine);
	return target;
}

static struct sched_group *
sched_balance_find_dst_group(struct sched_domain *sd, struct task_struct *p, int this_cpu);

/*
 * sched_balance_find_dst_group_cpu - find the idlest CPU among the CPUs in the group.
 */
/*
 * sched_balance_find_dst_group_cpu - 在候选 CPU 组内优先选择合适空闲 CPU，
 * 否则选负载最轻者。
 */
static int
sched_balance_find_dst_group_cpu(struct sched_group *group, struct task_struct *p, int this_cpu)
{
	unsigned long load, min_load = ULONG_MAX;
	unsigned int min_exit_latency = UINT_MAX;
	u64 latest_idle_timestamp = 0;
	int least_loaded_cpu = this_cpu;
	int shallowest_idle_cpu = -1;
	int i;

	/* Check if we have any choice: */
	if (group->group_weight == 1)
		return cpumask_first(sched_group_span(group));

	/* Traverse only the allowed CPUs */
	for_each_cpu_and(i, sched_group_span(group), p->cpus_ptr) {
		struct rq *rq = cpu_rq(i);

		if (!sched_core_cookie_match(rq, p))
			continue;

		if (sched_idle_cpu(i))
			return i;

		if (available_idle_cpu(i)) {
			struct cpuidle_state *idle = idle_get_state(rq);
			if (idle && idle->exit_latency < min_exit_latency) {
				/*
				 * We give priority to a CPU whose idle state
				 * has the smallest exit latency irrespective
				 * of any idle timestamp.
				 */
				min_exit_latency = idle->exit_latency;
				latest_idle_timestamp = rq->idle_stamp;
				shallowest_idle_cpu = i;
			} else if ((!idle || idle->exit_latency == min_exit_latency) &&
				   rq->idle_stamp > latest_idle_timestamp) {
				/*
				 * If equal or no active idle state, then
				 * the most recently idled CPU might have
				 * a warmer cache.
				 */
				latest_idle_timestamp = rq->idle_stamp;
				shallowest_idle_cpu = i;
			}
		} else if (shallowest_idle_cpu == -1) {
			load = cpu_load(cpu_rq(i));
			if (load < min_load) {
				min_load = load;
				least_loaded_cpu = i;
			}
		}
	}

	return shallowest_idle_cpu != -1 ? shallowest_idle_cpu : least_loaded_cpu;
}

/*
 * sched_balance_find_dst_cpu - 沿调度域逐级选择较空闲组及 CPU，
 * 细化任务的目标位置。
 */
static inline int sched_balance_find_dst_cpu(struct sched_domain *sd, struct task_struct *p,
				  int cpu, int prev_cpu, int sd_flag)
{
	int new_cpu = cpu;

	if (!cpumask_intersects(sched_domain_span(sd), p->cpus_ptr))
		return prev_cpu;

	/*
	 * We need task's util for cpu_util_without, sync it up to
	 * prev_cpu's last_update_time.
	 */
	if (!(sd_flag & SD_BALANCE_FORK))
		sync_entity_load_avg(&p->se);

	while (sd) {
		struct sched_group *group;
		struct sched_domain *tmp;
		int weight;

		if (!(sd->flags & sd_flag)) {
			sd = sd->child;
			continue;
		}

		group = sched_balance_find_dst_group(sd, p, cpu);
		if (!group) {
			sd = sd->child;
			continue;
		}

		new_cpu = sched_balance_find_dst_group_cpu(group, p, cpu);
		if (new_cpu == cpu) {
			/* Now try balancing at a lower domain level of 'cpu': */
			sd = sd->child;
			continue;
		}

		/* Now try balancing at a lower domain level of 'new_cpu': */
		cpu = new_cpu;
		weight = sd->span_weight;
		sd = NULL;
		for_each_domain(cpu, tmp) {
			if (weight <= tmp->span_weight)
				break;
			if (tmp->flags & sd_flag)
				sd = tmp;
		}
	}

	return new_cpu;
}

/*
 * __select_idle_cpu - 检查 CPU 是否空闲或仅有空闲策略任务，
 * 并满足核心 cookie 条件。
 */
static inline int __select_idle_cpu(int cpu, struct task_struct *p)
{
	if ((available_idle_cpu(cpu) || sched_idle_cpu(cpu)) &&
	    sched_cpu_cookie_match(cpu_rq(cpu), p))
		return cpu;

	return -1;
}

#ifdef CONFIG_SCHED_SMT
DEFINE_STATIC_KEY_FALSE(sched_smt_present);
EXPORT_SYMBOL_GPL(sched_smt_present);

/*
 * set_idle_cores - 更新共享缓存域中“可能存在整核空闲”的提示。
 */
static inline void set_idle_cores(int cpu, int val)
{
	struct sched_domain_shared *sds;

	sds = rcu_dereference(per_cpu(sd_llc_shared, cpu));
	if (sds)
		WRITE_ONCE(sds->has_idle_cores, val);
}

/*
 * test_idle_cores - 读取共享缓存域的空闲核心提示，减少无效的整核扫描。
 */
static inline bool test_idle_cores(int cpu)
{
	struct sched_domain_shared *sds;

	sds = rcu_dereference(per_cpu(sd_llc_shared, cpu));
	if (sds)
		return READ_ONCE(sds->has_idle_cores);

	return false;
}

/*
 * Scans the local SMT mask to see if the entire core is idle, and records this
 * information in sd_llc_shared->has_idle_cores.
 *
 * Since SMT siblings share all cache levels, inspecting this limited remote
 * state should be fairly cheap.
 */
/*
 * __update_idle_core - CPU 进入空闲时检查全部 SMT 兄弟，
 * 并设置域内存在空闲核心的提示。
 */
void __update_idle_core(struct rq *rq)
{
	int core = cpu_of(rq);
	int cpu;

	rcu_read_lock();
	if (test_idle_cores(core))
		goto unlock;

	for_each_cpu(cpu, cpu_smt_mask(core)) {
		if (cpu == core)
			continue;

		if (!available_idle_cpu(cpu))
			goto unlock;
	}

	set_idle_cores(core, 1);
unlock:
	rcu_read_unlock();
}

/*
 * Scan the entire LLC domain for idle cores; this dynamically switches off if
 * there are no idle cores left in the system; tracked through
 * sd_llc->shared->has_idle_cores and enabled through update_idle_core() above.
 */
/*
 * select_idle_core - 检查一个物理核心是否全空闲，并记录可用线程作为退路。
 */
static int select_idle_core(struct task_struct *p, int core, struct cpumask *cpus, int *idle_cpu)
{
	bool idle = true;
	int cpu;

	for_each_cpu(cpu, cpu_smt_mask(core)) {
		if (!available_idle_cpu(cpu)) {
			idle = false;
			if (*idle_cpu == -1) {
				if (sched_idle_cpu(cpu) && cpumask_test_cpu(cpu, cpus)) {
					*idle_cpu = cpu;
					break;
				}
				continue;
			}
			break;
		}
		if (*idle_cpu == -1 && cpumask_test_cpu(cpu, cpus))
			*idle_cpu = cpu;
	}

	if (idle)
		return core;

	cpumask_andnot(cpus, cpus, cpu_smt_mask(core));
	return -1;
}

/*
 * Scan the local SMT mask for idle CPUs.
 */
/*
 * select_idle_smt - 在目标核心的其他 SMT 线程中寻找亲和性允许的空闲 CPU。
 */
static int select_idle_smt(struct task_struct *p, struct sched_domain *sd, int target)
{
	int cpu;

	for_each_cpu_and(cpu, cpu_smt_mask(target), p->cpus_ptr) {
		if (cpu == target)
			continue;
		/*
		 * Check if the CPU is in the LLC scheduling domain of @target.
		 * Due to isolcpus, there is no guarantee that all the siblings are in the domain.
		 */
		if (!cpumask_test_cpu(cpu, sched_domain_span(sd)))
			continue;
		if (available_idle_cpu(cpu) || sched_idle_cpu(cpu))
			return cpu;
	}

	return -1;
}

#else /* !CONFIG_SCHED_SMT: */

/*
 * set_idle_cores - 未启用 SMT 时，无需维护整核空闲提示。
 */
static inline void set_idle_cores(int cpu, int val)
{
}

/*
 * test_idle_cores - 未启用 SMT 时，不使用额外的整核空闲提示。
 */
static inline bool test_idle_cores(int cpu)
{
	return false;
}

/*
 * select_idle_core - 未启用 SMT 时，整核选择退化为单个 CPU 的空闲检查。
 */
static inline int select_idle_core(struct task_struct *p, int core, struct cpumask *cpus, int *idle_cpu)
{
	return __select_idle_cpu(core, p);
}

/*
 * select_idle_smt - 未启用 SMT 时，无兄弟硬件线程可供选择。
 */
static inline int select_idle_smt(struct task_struct *p, struct sched_domain *sd, int target)
{
	return -1;
}

#endif /* !CONFIG_SCHED_SMT */

/*
 * Scan the LLC domain for idle CPUs; this is dynamically regulated by
 * comparing the average scan cost (tracked in sd->avg_scan_cost) against the
 * average idle time for this rq (as found in rq->avg_idle).
 */
/*
 * select_idle_cpu - 在调度域内按扫描预算寻找空闲 CPU，
 * 存在提示时优先寻找整核空闲。
 */
static int select_idle_cpu(struct task_struct *p, struct sched_domain *sd, bool has_idle_core, int target)
{
	struct cpumask *cpus = this_cpu_cpumask_var_ptr(select_rq_mask);
	int i, cpu, idle_cpu = -1, nr = INT_MAX;
	struct sched_domain_shared *sd_share;

	cpumask_and(cpus, sched_domain_span(sd), p->cpus_ptr);

	if (sched_feat(SIS_UTIL)) {
		sd_share = rcu_dereference(per_cpu(sd_llc_shared, target));
		if (sd_share) {
			/* because !--nr is the condition to stop scan */
			nr = READ_ONCE(sd_share->nr_idle_scan) + 1;
			/* overloaded LLC is unlikely to have idle cpu/core */
			if (nr == 1)
				return -1;
		}
	}

	if (static_branch_unlikely(&sched_cluster_active)) {
		struct sched_group *sg = sd->groups;

		if (sg->flags & SD_CLUSTER) {
			for_each_cpu_wrap(cpu, sched_group_span(sg), target + 1) {
				if (!cpumask_test_cpu(cpu, cpus))
					continue;

				if (has_idle_core) {
					i = select_idle_core(p, cpu, cpus, &idle_cpu);
					if ((unsigned int)i < nr_cpumask_bits)
						return i;
				} else {
					if (--nr <= 0)
						return -1;
					idle_cpu = __select_idle_cpu(cpu, p);
					if ((unsigned int)idle_cpu < nr_cpumask_bits)
						return idle_cpu;
				}
			}
			cpumask_andnot(cpus, cpus, sched_group_span(sg));
		}
	}

	for_each_cpu_wrap(cpu, cpus, target + 1) {
		if (has_idle_core) {
			i = select_idle_core(p, cpu, cpus, &idle_cpu);
			if ((unsigned int)i < nr_cpumask_bits)
				return i;

		} else {
			if (--nr <= 0)
				return -1;
			idle_cpu = __select_idle_cpu(cpu, p);
			if ((unsigned int)idle_cpu < nr_cpumask_bits)
				break;
		}
	}

	if (has_idle_core)
		set_idle_cores(target, false);

	return idle_cpu;
}

/*
 * Scan the asym_capacity domain for idle CPUs; pick the first idle one on which
 * the task fits. If no CPU is big enough, but there are idle ones, try to
 * maximize capacity.
 */
/*
 * select_idle_capacity - 在非对称容量域寻找适合任务的空闲 CPU，
 * 找不到时保留容量较佳候选。
 */
static int
select_idle_capacity(struct task_struct *p, struct sched_domain *sd, int target)
{
	unsigned long task_util, util_min, util_max, best_cap = 0;
	int fits, best_fits = 0;
	int cpu, best_cpu = -1;
	struct cpumask *cpus;

	cpus = this_cpu_cpumask_var_ptr(select_rq_mask);
	cpumask_and(cpus, sched_domain_span(sd), p->cpus_ptr);

	task_util = task_util_est(p);
	util_min = uclamp_eff_value(p, UCLAMP_MIN);
	util_max = uclamp_eff_value(p, UCLAMP_MAX);

	for_each_cpu_wrap(cpu, cpus, target) {
		unsigned long cpu_cap = capacity_of(cpu);

		if (!available_idle_cpu(cpu) && !sched_idle_cpu(cpu))
			continue;

		fits = util_fits_cpu(task_util, util_min, util_max, cpu);

		/* This CPU fits with all requirements */
		if (fits > 0)
			return cpu;
		/*
		 * Only the min performance hint (i.e. uclamp_min) doesn't fit.
		 * Look for the CPU with best capacity.
		 */
		else if (fits < 0)
			cpu_cap = get_actual_cpu_capacity(cpu);

		/*
		 * First, select CPU which fits better (-1 being better than 0).
		 * Then, select the one with best capacity at same level.
		 */
		if ((fits < best_fits) ||
		    ((fits == best_fits) && (cpu_cap > best_cap))) {
			best_cap = cpu_cap;
			best_cpu = cpu;
			best_fits = fits;
		}
	}

	return best_cpu;
}

/*
 * asym_fits_cpu - 仅在 CPU 容量不对称时执行任务与 CPU 的容量匹配检查。
 */
static inline bool asym_fits_cpu(unsigned long util,
				 unsigned long util_min,
				 unsigned long util_max,
				 int cpu)
{
	if (sched_asym_cpucap_active())
		/*
		 * Return true only if the cpu fully fits the task requirements
		 * which include the utilization and the performance hints.
		 */
		return (util_fits_cpu(util, util_min, util_max, cpu) > 0);

	return true;
}

/*
 * Try and locate an idle core/thread in the LLC cache domain.
 */
/*
 * select_idle_sibling - 从目标、上次和近期 CPU 开始寻找邻近空闲位置，
 * 再按拓扑扩大搜索。
 */
static int select_idle_sibling(struct task_struct *p, int prev, int target)
{
	bool has_idle_core = false;
	struct sched_domain *sd;
	unsigned long task_util, util_min, util_max;
	int i, recent_used_cpu, prev_aff = -1;

	/*
	 * On asymmetric system, update task utilization because we will check
	 * that the task fits with CPU's capacity.
	 */
	if (sched_asym_cpucap_active()) {
		sync_entity_load_avg(&p->se);
		task_util = task_util_est(p);
		util_min = uclamp_eff_value(p, UCLAMP_MIN);
		util_max = uclamp_eff_value(p, UCLAMP_MAX);
	}

	/*
	 * per-cpu select_rq_mask usage
	 */
	lockdep_assert_irqs_disabled();

	if ((available_idle_cpu(target) || sched_idle_cpu(target)) &&
	    asym_fits_cpu(task_util, util_min, util_max, target))
		return target;

	/*
	 * If the previous CPU is cache affine and idle, don't be stupid:
	 */
	if (prev != target && cpus_share_cache(prev, target) &&
	    (available_idle_cpu(prev) || sched_idle_cpu(prev)) &&
	    asym_fits_cpu(task_util, util_min, util_max, prev)) {

		if (!static_branch_unlikely(&sched_cluster_active) ||
		    cpus_share_resources(prev, target))
			return prev;

		prev_aff = prev;
	}

	/*
	 * Allow a per-cpu kthread to stack with the wakee if the
	 * kworker thread and the tasks previous CPUs are the same.
	 * The assumption is that the wakee queued work for the
	 * per-cpu kthread that is now complete and the wakeup is
	 * essentially a sync wakeup. An obvious example of this
	 * pattern is IO completions.
	 */
	if (is_per_cpu_kthread(current) &&
	    in_task() &&
	    prev == smp_processor_id() &&
	    this_rq()->nr_running <= 1 &&
	    asym_fits_cpu(task_util, util_min, util_max, prev)) {
		return prev;
	}

	/* Check a recently used CPU as a potential idle candidate: */
	recent_used_cpu = p->recent_used_cpu;
	p->recent_used_cpu = prev;
	if (recent_used_cpu != prev &&
	    recent_used_cpu != target &&
	    cpus_share_cache(recent_used_cpu, target) &&
	    (available_idle_cpu(recent_used_cpu) || sched_idle_cpu(recent_used_cpu)) &&
	    cpumask_test_cpu(recent_used_cpu, p->cpus_ptr) &&
	    asym_fits_cpu(task_util, util_min, util_max, recent_used_cpu)) {

		if (!static_branch_unlikely(&sched_cluster_active) ||
		    cpus_share_resources(recent_used_cpu, target))
			return recent_used_cpu;

	} else {
		recent_used_cpu = -1;
	}

	/*
	 * For asymmetric CPU capacity systems, our domain of interest is
	 * sd_asym_cpucapacity rather than sd_llc.
	 */
	if (sched_asym_cpucap_active()) {
		sd = rcu_dereference(per_cpu(sd_asym_cpucapacity, target));
		/*
		 * On an asymmetric CPU capacity system where an exclusive
		 * cpuset defines a symmetric island (i.e. one unique
		 * capacity_orig value through the cpuset), the key will be set
		 * but the CPUs within that cpuset will not have a domain with
		 * SD_ASYM_CPUCAPACITY. These should follow the usual symmetric
		 * capacity path.
		 */
		if (sd) {
			i = select_idle_capacity(p, sd, target);
			return ((unsigned)i < nr_cpumask_bits) ? i : target;
		}
	}

	sd = rcu_dereference(per_cpu(sd_llc, target));
	if (!sd)
		return target;

	if (sched_smt_active()) {
		has_idle_core = test_idle_cores(target);

		if (!has_idle_core && cpus_share_cache(prev, target)) {
			i = select_idle_smt(p, sd, prev);
			if ((unsigned int)i < nr_cpumask_bits)
				return i;
		}
	}

	i = select_idle_cpu(p, sd, has_idle_core, target);
	if ((unsigned)i < nr_cpumask_bits)
		return i;

	/*
	 * For cluster machines which have lower sharing cache like L2 or
	 * LLC Tag, we tend to find an idle CPU in the target's cluster
	 * first. But prev_cpu or recent_used_cpu may also be a good candidate,
	 * use them if possible when no idle CPU found in select_idle_cpu().
	 */
	if ((unsigned int)prev_aff < nr_cpumask_bits)
		return prev_aff;
	if ((unsigned int)recent_used_cpu < nr_cpumask_bits)
		return recent_used_cpu;

	return target;
}

/*
 * cpu_util - 估算指定 CPU 的公平利用率，并可模拟任务迁入或迁出后的状态。
 * 按需合并 runnable 需求及 UTIL_EST 预测，最终限制在架构容量以内。
 */
/**
 * cpu_util() - Estimates the amount of CPU capacity used by CFS tasks.
 * @cpu: the CPU to get the utilization for
 * @p: task for which the CPU utilization should be predicted or NULL
 * @dst_cpu: CPU @p migrates to, -1 if @p moves from @cpu or @p == NULL
 * @boost: 1 to enable boosting, otherwise 0
 *
 * The unit of the return value must be the same as the one of CPU capacity
 * so that CPU utilization can be compared with CPU capacity.
 *
 * CPU utilization is the sum of running time of runnable tasks plus the
 * recent utilization of currently non-runnable tasks on that CPU.
 * It represents the amount of CPU capacity currently used by CFS tasks in
 * the range [0..max CPU capacity] with max CPU capacity being the CPU
 * capacity at f_max.
 *
 * The estimated CPU utilization is defined as the maximum between CPU
 * utilization and sum of the estimated utilization of the currently
 * runnable tasks on that CPU. It preserves a utilization "snapshot" of
 * previously-executed tasks, which helps better deduce how busy a CPU will
 * be when a long-sleeping task wakes up. The contribution to CPU utilization
 * of such a task would be significantly decayed at this point of time.
 *
 * Boosted CPU utilization is defined as max(CPU runnable, CPU utilization).
 * CPU contention for CFS tasks can be detected by CPU runnable > CPU
 * utilization. Boosting is implemented in cpu_util() so that internal
 * users (e.g. EAS) can use it next to external users (e.g. schedutil),
 * latter via cpu_util_cfs_boost().
 *
 * CPU utilization can be higher than the current CPU capacity
 * (f_curr/f_max * max CPU capacity) or even the max CPU capacity because
 * of rounding errors as well as task migrations or wakeups of new tasks.
 * CPU utilization has to be capped to fit into the [0..max CPU capacity]
 * range. Otherwise a group of CPUs (CPU0 util = 121% + CPU1 util = 80%)
 * could be seen as over-utilized even though CPU1 has 20% of spare CPU
 * capacity. CPU utilization is allowed to overshoot current CPU capacity
 * though since this is useful for predicting the CPU capacity required
 * after task migrations (scheduler-driven DVFS).
 *
 * Return: (Boosted) (estimated) utilization for the specified CPU.
 */
static unsigned long
cpu_util(int cpu, struct task_struct *p, int dst_cpu, int boost)
{
	bool add_task = p && task_cpu(p) != cpu && dst_cpu == cpu;
	bool sub_task = p && task_cpu(p) == cpu && dst_cpu != cpu;
	struct cfs_rq *cfs_rq = &cpu_rq(cpu)->cfs;
	unsigned long util = READ_ONCE(cfs_rq->avg.util_avg);
	unsigned long runnable;

	/*
	 * If @dst_cpu is -1 or @p migrates from @cpu to @dst_cpu remove its
	 * contribution. If @p migrates from another CPU to @cpu add its
	 * contribution. In all the other cases @cpu is not impacted by the
	 * migration so its util_avg is already correct.
	 */
	if (add_task)
		util += task_util(p);
	else if (sub_task)
		lsub_positive(&util, task_util(p));

	if (boost) {
		runnable = READ_ONCE(cfs_rq->avg.runnable_avg);
		if (add_task)
			runnable += READ_ONCE(p->se.avg.runnable_avg);
		else if (sub_task)
			lsub_positive(&runnable,
				      READ_ONCE(p->se.avg.runnable_avg));
		util = max(util, runnable);
	}

	if (sched_feat(UTIL_EST)) {
		unsigned long util_est;

		util_est = READ_ONCE(cfs_rq->avg.util_est);

		/*
		 * During wake-up @p isn't enqueued yet and doesn't contribute
		 * to any cpu_rq(cpu)->cfs.avg.util_est.
		 * If @dst_cpu == @cpu add it to "simulate" cpu_util after @p
		 * has been enqueued.
		 *
		 * During exec (@dst_cpu = -1) @p is enqueued and does
		 * contribute to cpu_rq(cpu)->cfs.util_est.
		 * Remove it to "simulate" cpu_util without @p's contribution.
		 *
		 * Despite the task_on_rq_queued(@p) check there is still a
		 * small window for a possible race when an exec
		 * select_task_rq_fair() races with LB's detach_task().
		 *
		 *   detach_task()
		 *     deactivate_task()
		 *       p->on_rq = TASK_ON_RQ_MIGRATING;
		 *       -------------------------------- A
		 *       dequeue_task()                    \
		 *         dequeue_task_fair()              + Race Time
		 *           util_est_dequeue()            /
		 *       -------------------------------- B
		 *
		 * The additional check "current == p" is required to further
		 * reduce the race window.
		 */
		if (dst_cpu == cpu)
			util_est += _task_util_est(p);
		else if (p && unlikely(task_on_rq_queued(p) || current == p))
			lsub_positive(&util_est, _task_util_est(p));

		util = max(util, util_est);
	}

	return min(util, arch_scale_cpu_capacity(cpu));
}

/*
 * cpu_util_cfs - 取得 CPU 当前公平负载的利用率估计。
 */
unsigned long cpu_util_cfs(int cpu)
{
	return cpu_util(cpu, NULL, -1, 0);
}

/*
 * cpu_util_cfs_boost - 把等待执行的需求也计入利用率，
 * 供需要提高性能的路径使用。
 */
unsigned long cpu_util_cfs_boost(int cpu)
{
	return cpu_util(cpu, NULL, -1, 1);
}

/*
 * cpu_util_without: compute cpu utilization without any contributions from *p
 * @cpu: the CPU which utilization is requested
 * @p: the task which utilization should be discounted
 *
 * The utilization of a CPU is defined by the utilization of tasks currently
 * enqueued on that CPU as well as tasks which are currently sleeping after an
 * execution on that CPU.
 *
 * This method returns the utilization of the specified CPU by discounting the
 * utilization of the specified task, whenever the task is currently
 * contributing to the CPU utilization.
 */
/*
 * cpu_util_without - 估算去掉指定任务后的公平利用率，
 * 作为选核和能耗比较基线。
 */
static unsigned long cpu_util_without(int cpu, struct task_struct *p)
{
	/* Task has no contribution or is new */
	if (cpu != task_cpu(p) || !READ_ONCE(p->se.avg.last_update_time))
		p = NULL;

	return cpu_util(cpu, p, -1, 0);
}

/*
 * This function computes an effective utilization for the given CPU, to be
 * used for frequency selection given the linear relation: f = u * f_max.
 *
 * The scheduler tracks the following metrics:
 *
 *   cpu_util_{cfs,rt,dl,irq}()
 *   cpu_bw_dl()
 *
 * Where the cfs,rt and dl util numbers are tracked with the same metric and
 * synchronized windows and are thus directly comparable.
 *
 * The cfs,rt,dl utilization are the running times measured with rq->clock_task
 * which excludes things like IRQ and steal-time. These latter are then accrued
 * in the IRQ utilization.
 *
 * The DL bandwidth number OTOH is not a measured metric but a value computed
 * based on the task model parameters and gives the minimal utilization
 * required to meet deadlines.
 */
/*
 * effective_cpu_util - 合并公平、实时、截止期和中断负载，
 * 得到 CPU 的有效利用率。
 * 可同时输出包含 uclamp 和截止期带宽要求的性能上下界。
 */
unsigned long effective_cpu_util(int cpu, unsigned long util_cfs,
				 unsigned long *min,
				 unsigned long *max)
{
	unsigned long util, irq, scale;
	struct rq *rq = cpu_rq(cpu);

	scale = arch_scale_cpu_capacity(cpu);

	/*
	 * Early check to see if IRQ/steal time saturates the CPU, can be
	 * because of inaccuracies in how we track these -- see
	 * update_irq_load_avg().
	 */
	irq = cpu_util_irq(rq);
	if (unlikely(irq >= scale)) {
		if (min)
			*min = scale;
		if (max)
			*max = scale;
		return scale;
	}

	if (min) {
		/*
		 * The minimum utilization returns the highest level between:
		 * - the computed DL bandwidth needed with the IRQ pressure which
		 *   steals time to the deadline task.
		 * - The minimum performance requirement for CFS and/or RT.
		 */
		*min = max(irq + cpu_bw_dl(rq), uclamp_rq_get(rq, UCLAMP_MIN));

		/*
		 * When an RT task is runnable and uclamp is not used, we must
		 * ensure that the task will run at maximum compute capacity.
		 */
		if (!uclamp_is_used() && rt_rq_is_runnable(&rq->rt))
			*min = max(*min, scale);
	}

	/*
	 * Because the time spend on RT/DL tasks is visible as 'lost' time to
	 * CFS tasks and we use the same metric to track the effective
	 * utilization (PELT windows are synchronized) we can directly add them
	 * to obtain the CPU's actual utilization.
	 */
	util = util_cfs + cpu_util_rt(rq);
	util += cpu_util_dl(rq);

	/*
	 * The maximum hint is a soft bandwidth requirement, which can be lower
	 * than the actual utilization because of uclamp_max requirements.
	 */
	if (max)
		*max = min(scale, uclamp_rq_get(rq, UCLAMP_MAX));

	if (util >= scale)
		return scale;

	/*
	 * There is still idle time; further improve the number by using the
	 * IRQ metric. Because IRQ/steal time is hidden from the task clock we
	 * need to scale the task numbers:
	 *
	 *              max - irq
	 *   U' = irq + --------- * U
	 *                 max
	 */
	util = scale_irq_capacity(util, irq, scale);
	util += irq;

	return min(scale, util);
}

/*
 * sched_cpu_util - 对外提供调度器估计的 CPU 总有效利用率。
 */
unsigned long sched_cpu_util(int cpu)
{
	return effective_cpu_util(cpu, cpu_util_cfs(cpu), NULL, NULL);
}

/*
 * energy_env - Utilization landscape for energy estimation.
 * @task_busy_time: Utilization contribution by the task for which we test the
 *                  placement. Given by eenv_task_busy_time().
 * @pd_busy_time:   Utilization of the whole perf domain without the task
 *                  contribution. Given by eenv_pd_busy_time().
 * @cpu_cap:        Maximum CPU capacity for the perf domain.
 * @pd_cap:         Entire perf domain capacity. (pd->nr_cpus * cpu_cap).
 */
struct energy_env {
	unsigned long task_busy_time;
	unsigned long pd_busy_time;
	unsigned long cpu_cap;
	unsigned long pd_cap;
};

/*
 * Compute the task busy time for compute_energy(). This time cannot be
 * injected directly into effective_cpu_util() because of the IRQ scaling.
 * The latter only makes sense with the most recent CPUs where the task has
 * run.
 */
/*
 * eenv_task_busy_time - 估算待放置任务的忙碌时间贡献，
 * 并补偿源 CPU 的中断占用。
 */
static inline void eenv_task_busy_time(struct energy_env *eenv,
				       struct task_struct *p, int prev_cpu)
{
	unsigned long busy_time, max_cap = arch_scale_cpu_capacity(prev_cpu);
	unsigned long irq = cpu_util_irq(cpu_rq(prev_cpu));

	if (unlikely(irq >= max_cap))
		busy_time = max_cap;
	else
		busy_time = scale_irq_capacity(task_util_est(p), irq, max_cap);

	eenv->task_busy_time = busy_time;
}

/*
 * Compute the perf_domain (PD) busy time for compute_energy(). Based on the
 * utilization for each @pd_cpus, it however doesn't take into account
 * clamping since the ratio (utilization / cpu_capacity) is already enough to
 * scale the EM reported power consumption at the (eventually clamped)
 * cpu_capacity.
 *
 * The contribution of the task @p for which we want to estimate the
 * energy cost is removed (by cpu_util()) and must be calculated
 * separately (see eenv_task_busy_time). This ensures:
 *
 *   - A stable PD utilization, no matter which CPU of that PD we want to place
 *     the task on.
 *
 *   - A fair comparison between CPUs as the task contribution (task_util())
 *     will always be the same no matter which CPU utilization we rely on
 *     (util_avg or util_est).
 *
 * Set @eenv busy time for the PD that spans @pd_cpus. This busy time can't
 * exceed @eenv->pd_cap.
 */
/*
 * eenv_pd_busy_time - 汇总性能域中排除待放置任务后的忙碌时间，
 * 作为能耗基线。
 */
static inline void eenv_pd_busy_time(struct energy_env *eenv,
				     struct cpumask *pd_cpus,
				     struct task_struct *p)
{
	unsigned long busy_time = 0;
	int cpu;

	for_each_cpu(cpu, pd_cpus) {
		unsigned long util = cpu_util(cpu, p, -1, 0);

		busy_time += effective_cpu_util(cpu, util, NULL, NULL);
	}

	eenv->pd_busy_time = min(eenv->pd_cap, busy_time);
}

/*
 * Compute the maximum utilization for compute_energy() when the task @p
 * is placed on the cpu @dst_cpu.
 *
 * Returns the maximum utilization among @eenv->cpus. This utilization can't
 * exceed @eenv->cpu_cap.
 */
/*
 * eenv_pd_max_util - 模拟任务放置后性能域内的最大性能需求，
 * 用于估算共享频率。
 */
static inline unsigned long
eenv_pd_max_util(struct energy_env *eenv, struct cpumask *pd_cpus,
		 struct task_struct *p, int dst_cpu)
{
	unsigned long max_util = 0;
	int cpu;

	for_each_cpu(cpu, pd_cpus) {
		struct task_struct *tsk = (cpu == dst_cpu) ? p : NULL;
		unsigned long util = cpu_util(cpu, p, dst_cpu, 1);
		unsigned long eff_util, min, max;

		/*
		 * Performance domain frequency: utilization clamping
		 * must be considered since it affects the selection
		 * of the performance domain frequency.
		 * NOTE: in case RT tasks are running, by default the min
		 * utilization can be max OPP.
		 */
		eff_util = effective_cpu_util(cpu, util, &min, &max);

		/* Task's uclamp can modify min and max value */
		if (tsk && uclamp_is_used()) {
			min = max(min, uclamp_eff_value(p, UCLAMP_MIN));

			/*
			 * If there is no active max uclamp constraint,
			 * directly use task's one, otherwise keep max.
			 */
			if (uclamp_rq_is_idle(cpu_rq(cpu)))
				max = uclamp_eff_value(p, UCLAMP_MAX);
			else
				max = max(max, uclamp_eff_value(p, UCLAMP_MAX));
		}

		eff_util = sugov_effective_cpu_perf(cpu, eff_util, min, max);
		max_util = max(max_util, eff_util);
	}

	return min(max_util, eenv->cpu_cap);
}

/*
 * compute_energy(): Use the Energy Model to estimate the energy that @pd would
 * consume for a given utilization landscape @eenv. When @dst_cpu < 0, the task
 * contribution is ignored.
 */
/*
 * compute_energy - 用性能域最大需求和总忙碌时间调用能量模型，
 * 估算给定放置方案的能耗。
 */
static inline unsigned long
compute_energy(struct energy_env *eenv, struct perf_domain *pd,
	       struct cpumask *pd_cpus, struct task_struct *p, int dst_cpu)
{
	unsigned long max_util = eenv_pd_max_util(eenv, pd_cpus, p, dst_cpu);
	unsigned long busy_time = eenv->pd_busy_time;
	unsigned long energy;

	if (dst_cpu >= 0)
		busy_time = min(eenv->pd_cap, busy_time + eenv->task_busy_time);

	energy = em_cpu_energy(pd->em_pd, max_util, busy_time, eenv->cpu_cap);

	trace_sched_compute_energy_tp(p, dst_cpu, energy, max_util, busy_time);

	return energy;
}

/*
 * find_energy_efficient_cpu(): Find most energy-efficient target CPU for the
 * waking task. find_energy_efficient_cpu() looks for the CPU with maximum
 * spare capacity in each performance domain and uses it as a potential
 * candidate to execute the task. Then, it uses the Energy Model to figure
 * out which of the CPU candidates is the most energy-efficient.
 *
 * The rationale for this heuristic is as follows. In a performance domain,
 * all the most energy efficient CPU candidates (according to the Energy
 * Model) are those for which we'll request a low frequency. When there are
 * several CPUs for which the frequency request will be the same, we don't
 * have enough data to break the tie between them, because the Energy Model
 * only includes active power costs. With this model, if we assume that
 * frequency requests follow utilization (e.g. using schedutil), the CPU with
 * the maximum spare capacity in a performance domain is guaranteed to be among
 * the best candidates of the performance domain.
 *
 * In practice, it could be preferable from an energy standpoint to pack
 * small tasks on a CPU in order to let other CPUs go in deeper idle states,
 * but that could also hurt our chances to go cluster idle, and we have no
 * ways to tell with the current Energy Model if this is actually a good
 * idea or not. So, find_energy_efficient_cpu() basically favors
 * cluster-packing, and spreading inside a cluster. That should at least be
 * a good thing for latency, and this is consistent with the idea that most
 * of the energy savings of EAS come from the asymmetry of the system, and
 * not so much from breaking the tie between identical CPUs. That's also the
 * reason why EAS is enabled in the topology code only for systems where
 * SD_ASYM_CPUCAPACITY is set.
 *
 * NOTE: Forkees are not accepted in the energy-aware wake-up path because
 * they don't have any useful utilization data yet and it's not possible to
 * forecast their impact on energy consumption. Consequently, they will be
 * placed by sched_balance_find_dst_cpu() on the least loaded CPU, which might turn out
 * to be energy-inefficient in some use-cases. The alternative would be to
 * bias new tasks towards specific types of CPUs first, or to try to infer
 * their util_avg from the parent task, but those heuristics could hurt
 * other use-cases too. So, until someone finds a better way to solve this,
 * let's keep things simple by re-using the existing slow path.
 */
/*
 * find_energy_efficient_cpu：
 * 在可用性能域中比较任务放置带来的能耗增量及容量匹配程度。
 * 模型或拓扑不适用时返回失败或保留原 CPU，由调用者继续常规选择。
 */
static int find_energy_efficient_cpu(struct task_struct *p, int prev_cpu)
{
	struct cpumask *cpus = this_cpu_cpumask_var_ptr(select_rq_mask);
	unsigned long prev_delta = ULONG_MAX, best_delta = ULONG_MAX;
	unsigned long p_util_min = uclamp_is_used() ? uclamp_eff_value(p, UCLAMP_MIN) : 0;
	unsigned long p_util_max = uclamp_is_used() ? uclamp_eff_value(p, UCLAMP_MAX) : 1024;
	struct root_domain *rd = this_rq()->rd;
	int cpu, best_energy_cpu, target = -1;
	int prev_fits = -1, best_fits = -1;
	unsigned long best_actual_cap = 0;
	unsigned long prev_actual_cap = 0;
	struct sched_domain *sd;
	struct perf_domain *pd;
	struct energy_env eenv;

	rcu_read_lock();
	pd = rcu_dereference(rd->pd);
	if (!pd)
		goto unlock;

	/*
	 * Energy-aware wake-up happens on the lowest sched_domain starting
	 * from sd_asym_cpucapacity spanning over this_cpu and prev_cpu.
	 */
	sd = rcu_dereference(*this_cpu_ptr(&sd_asym_cpucapacity));
	while (sd && !cpumask_test_cpu(prev_cpu, sched_domain_span(sd)))
		sd = sd->parent;
	if (!sd)
		goto unlock;

	target = prev_cpu;

	sync_entity_load_avg(&p->se);
	if (!task_util_est(p) && p_util_min == 0)
		goto unlock;

	eenv_task_busy_time(&eenv, p, prev_cpu);

	for (; pd; pd = pd->next) {
		unsigned long util_min = p_util_min, util_max = p_util_max;
		unsigned long cpu_cap, cpu_actual_cap, util;
		long prev_spare_cap = -1, max_spare_cap = -1;
		unsigned long rq_util_min, rq_util_max;
		unsigned long cur_delta, base_energy;
		int max_spare_cap_cpu = -1;
		int fits, max_fits = -1;

		cpumask_and(cpus, perf_domain_span(pd), cpu_online_mask);

		if (cpumask_empty(cpus))
			continue;

		/* Account external pressure for the energy estimation */
		cpu = cpumask_first(cpus);
		cpu_actual_cap = get_actual_cpu_capacity(cpu);

		eenv.cpu_cap = cpu_actual_cap;
		eenv.pd_cap = 0;

		for_each_cpu(cpu, cpus) {
			struct rq *rq = cpu_rq(cpu);

			eenv.pd_cap += cpu_actual_cap;

			if (!cpumask_test_cpu(cpu, sched_domain_span(sd)))
				continue;

			if (!cpumask_test_cpu(cpu, p->cpus_ptr))
				continue;

			util = cpu_util(cpu, p, cpu, 0);
			cpu_cap = capacity_of(cpu);

			/*
			 * Skip CPUs that cannot satisfy the capacity request.
			 * IOW, placing the task there would make the CPU
			 * overutilized. Take uclamp into account to see how
			 * much capacity we can get out of the CPU; this is
			 * aligned with sched_cpu_util().
			 */
			if (uclamp_is_used() && !uclamp_rq_is_idle(rq)) {
				/*
				 * Open code uclamp_rq_util_with() except for
				 * the clamp() part. I.e.: apply max aggregation
				 * only. util_fits_cpu() logic requires to
				 * operate on non clamped util but must use the
				 * max-aggregated uclamp_{min, max}.
				 */
				rq_util_min = uclamp_rq_get(rq, UCLAMP_MIN);
				rq_util_max = uclamp_rq_get(rq, UCLAMP_MAX);

				util_min = max(rq_util_min, p_util_min);
				util_max = max(rq_util_max, p_util_max);
			}

			fits = util_fits_cpu(util, util_min, util_max, cpu);
			if (!fits)
				continue;

			lsub_positive(&cpu_cap, util);

			if (cpu == prev_cpu) {
				/* Always use prev_cpu as a candidate. */
				prev_spare_cap = cpu_cap;
				prev_fits = fits;
			} else if ((fits > max_fits) ||
				   ((fits == max_fits) && ((long)cpu_cap > max_spare_cap))) {
				/*
				 * Find the CPU with the maximum spare capacity
				 * among the remaining CPUs in the performance
				 * domain.
				 */
				max_spare_cap = cpu_cap;
				max_spare_cap_cpu = cpu;
				max_fits = fits;
			}
		}

		if (max_spare_cap_cpu < 0 && prev_spare_cap < 0)
			continue;

		eenv_pd_busy_time(&eenv, cpus, p);
		/* Compute the 'base' energy of the pd, without @p */
		/*
		 * 先算不含任务的基线，再比较加入任务的增量；
		 * 性能域内 CPU 共享频率影响。
		 */
		base_energy = compute_energy(&eenv, pd, cpus, p, -1);

		/* Evaluate the energy impact of using prev_cpu. */
		if (prev_spare_cap > -1) {
			prev_delta = compute_energy(&eenv, pd, cpus, p,
						    prev_cpu);
			/* CPU utilization has changed */
			if (prev_delta < base_energy)
				goto unlock;
			prev_delta -= base_energy;
			prev_actual_cap = cpu_actual_cap;
			best_delta = min(best_delta, prev_delta);
		}

		/* Evaluate the energy impact of using max_spare_cap_cpu. */
		if (max_spare_cap_cpu >= 0 && max_spare_cap > prev_spare_cap) {
			/* Current best energy cpu fits better */
			if (max_fits < best_fits)
				continue;

			/*
			 * Both don't fit performance hint (i.e. uclamp_min)
			 * but best energy cpu has better capacity.
			 */
			if ((max_fits < 0) &&
			    (cpu_actual_cap <= best_actual_cap))
				continue;

			cur_delta = compute_energy(&eenv, pd, cpus, p,
						   max_spare_cap_cpu);
			/* CPU utilization has changed */
			if (cur_delta < base_energy)
				goto unlock;
			cur_delta -= base_energy;

			/*
			 * Both fit for the task but best energy cpu has lower
			 * energy impact.
			 */
			if ((max_fits > 0) && (best_fits > 0) &&
			    (cur_delta >= best_delta))
				continue;

			best_delta = cur_delta;
			best_energy_cpu = max_spare_cap_cpu;
			best_fits = max_fits;
			best_actual_cap = cpu_actual_cap;
		}
	}
	rcu_read_unlock();

	if ((best_fits > prev_fits) ||
	    ((best_fits > 0) && (best_delta < prev_delta)) ||
	    ((best_fits < 0) && (best_actual_cap > prev_actual_cap)))
		target = best_energy_cpu;

	return target;

unlock:
	rcu_read_unlock();

	return target;
}

/*
 * select_task_rq_fair: Select target runqueue for the waking task in domains
 * that have the relevant SD flag set. In practice, this is SD_BALANCE_WAKE,
 * SD_BALANCE_FORK, or SD_BALANCE_EXEC.
 *
 * Balances load by selecting the idlest CPU in the idlest group, or under
 * certain conditions an idle sibling CPU if the domain has SD_WAKE_AFFINE set.
 *
 * Returns the target CPU number.
 */
/*
 * select_task_rq_fair - 唤醒、fork 或 exec 时为公平任务选择目标 CPU。
 * 依次结合显式唤醒提示、能耗策略、唤醒亲和性和空闲/负载搜索。
 */
static int
select_task_rq_fair(struct task_struct *p, int prev_cpu, int wake_flags)
{
	int sync = (wake_flags & WF_SYNC) && !(current->flags & PF_EXITING);
	struct sched_domain *tmp, *sd = NULL;
	int cpu = smp_processor_id();
	int new_cpu = prev_cpu;
	int want_affine = 0;
	/* SD_flags and WF_flags share the first nibble */
	int sd_flag = wake_flags & 0xF;

	/*
	 * required for stable ->cpus_allowed
	 */
	lockdep_assert_held(&p->pi_lock);
	/*
	 * 普通唤醒优先尝试提示和能耗选核；
	 * 这些快速决策不适用时再遍历调度域。
	 */
	if (wake_flags & WF_TTWU) {
		record_wakee(p);

		if ((wake_flags & WF_CURRENT_CPU) &&
		    cpumask_test_cpu(cpu, p->cpus_ptr))
			return cpu;

		if (!is_rd_overutilized(this_rq()->rd)) {
			new_cpu = find_energy_efficient_cpu(p, prev_cpu);
			if (new_cpu >= 0)
				return new_cpu;
			new_cpu = prev_cpu;
		}

		want_affine = !wake_wide(p) && cpumask_test_cpu(cpu, p->cpus_ptr);
	}

	rcu_read_lock();
	for_each_domain(cpu, tmp) {
		/*
		 * If both 'cpu' and 'prev_cpu' are part of this domain,
		 * cpu is a valid SD_WAKE_AFFINE target.
		 */
		if (want_affine && (tmp->flags & SD_WAKE_AFFINE) &&
		    cpumask_test_cpu(prev_cpu, sched_domain_span(tmp))) {
			if (cpu != prev_cpu)
				new_cpu = wake_affine(tmp, p, cpu, prev_cpu, sync);

			sd = NULL; /* Prefer wake_affine over balance flags */
			break;
		}

		/*
		 * Usually only true for WF_EXEC and WF_FORK, as sched_domains
		 * usually do not have SD_BALANCE_WAKE set. That means wakeup
		 * will usually go to the fast path.
		 */
		if (tmp->flags & sd_flag)
			sd = tmp;
		else if (!want_affine)
			break;
	}

	if (unlikely(sd)) {
		/* Slow path */
		new_cpu = sched_balance_find_dst_cpu(sd, p, cpu, prev_cpu, sd_flag);
	} else if (wake_flags & WF_TTWU) { /* XXX always ? */
		/* Fast path */
		new_cpu = select_idle_sibling(p, prev_cpu, new_cpu);
	}
	rcu_read_unlock();

	return new_cpu;
}

/*
 * Called immediately before a task is migrated to a new CPU; task_cpu(p) and
 * cfs_rq_of(p) references at time of call are still valid and identify the
 * previous CPU. The caller guarantees p->pi_lock or task_rq(p)->lock is held.
 */
/*
 * migrate_task_rq_fair - 迁移前撤销源队列历史贡献并重置 PELT 时间戳，
 * 使目标队列重新接入。
 */
static void migrate_task_rq_fair(struct task_struct *p, int new_cpu)
{
	struct sched_entity *se = &p->se;

	if (!task_on_rq_migrating(p)) {
		remove_entity_load_avg(se);

		/*
		 * Here, the task's PELT values have been updated according to
		 * the current rq's clock. But if that clock hasn't been
		 * updated in a while, a substantial idle time will be missed,
		 * leading to an inflation after wake-up on the new rq.
		 *
		 * Estimate the missing time from the cfs_rq last_update_time
		 * and update sched_avg to improve the PELT continuity after
		 * migration.
		 */
		migrate_se_pelt_lag(se);
	}

	/* Tell new CPU we are migrated */
	se->avg.last_update_time = 0;

	update_scan_period(p, new_cpu);
}

/*
 * task_dead_fair - 任务退出时完成尚未结束的延迟出队，
 * 并移除其历史负载贡献。
 */
static void task_dead_fair(struct task_struct *p)
{
	struct sched_entity *se = &p->se;

	if (se->sched_delayed) {
		struct rq_flags rf;
		struct rq *rq;

		rq = task_rq_lock(p, &rf);
		if (se->sched_delayed) {
			update_rq_clock(rq);
			dequeue_entities(rq, se, DEQUEUE_SLEEP | DEQUEUE_DELAYED);
		}
		task_rq_unlock(rq, p, &rf);
	}

	remove_entity_load_avg(se);
}

/*
 * Set the max capacity the task is allowed to run at for misfit detection.
 */
/*
 * set_task_max_allowed_capacity - 缓存任务亲和性允许范围中的最大 CPU 容量，
 * 供 misfit 判断使用。
 */
static void set_task_max_allowed_capacity(struct task_struct *p)
{
	struct asym_cap_data *entry;

	if (!sched_asym_cpucap_active())
		return;

	rcu_read_lock();
	list_for_each_entry_rcu(entry, &asym_cap_list, link) {
		cpumask_t *cpumask;

		cpumask = cpu_capacity_span(entry);
		if (!cpumask_intersects(p->cpus_ptr, cpumask))
			continue;

		p->max_allowed_capacity = entry->capacity;
		break;
	}
	rcu_read_unlock();
}

/*
 * set_cpus_allowed_fair - 应用 CPU 亲和性变更，并刷新允许的最大容量。
 */
static void set_cpus_allowed_fair(struct task_struct *p, struct affinity_context *ctx)
{
	set_cpus_allowed_common(p, ctx);
	set_task_max_allowed_capacity(p);
}

/*
 * balance_fair - 有公平任务时直接继续选择，
 * 否则尝试新空闲 CPU 的拉取均衡。
 */
static int
balance_fair(struct rq *rq, struct rq_flags *rf)
{
	if (sched_fair_runnable(rq))
		return 1;

	return sched_balance_newidle(rq, rf) != 0;
}

/*
 * set_next_buddy - 沿组层级设置下一候选偏好；
 * 不为未在队或空闲策略实体建立提示。
 */
static void set_next_buddy(struct sched_entity *se)
{
	for_each_sched_entity(se) {
		if (WARN_ON_ONCE(!se->on_rq))
			return;
		if (se_is_idle(se))
			return;
		cfs_rq_of(se)->next = se;
	}
}

enum preempt_wakeup_action {
	PREEMPT_WAKEUP_NONE,	/* No preemption. */
	PREEMPT_WAKEUP_SHORT,	/* Ignore slice protection. */
	PREEMPT_WAKEUP_PICK,	/* Let __pick_eevdf() decide. */
	PREEMPT_WAKEUP_RESCHED,	/* Force reschedule. */
};

/*
 * set_preempt_buddy - 提名被唤醒实体为 next buddy，
 * 同时保留截止时间更早的已有候选。
 */
static inline bool
set_preempt_buddy(struct cfs_rq *cfs_rq, int wake_flags,
		  struct sched_entity *pse, struct sched_entity *se)
{
	/*
	 * Keep existing buddy if the deadline is sooner than pse.
	 * The older buddy may be cache cold and completely unrelated
	 * to the current wakeup but that is unpredictable where as
	 * obeying the deadline is more in line with EEVDF objectives.
	 */
	if (cfs_rq->next && entity_before(cfs_rq->next, pse))
		return false;

	set_next_buddy(pse);
	return true;
}

/*
 * WF_SYNC|WF_TTWU indicates the waker expects to sleep but it is not
 * strictly enforced because the hint is either misunderstood or
 * multiple tasks must be woken up.
 */
/*
 * preempt_sync - 评估同步唤醒提示是否应请求抢占，
 * 给唤醒者保留少量批量唤醒时间。
 */
static inline enum preempt_wakeup_action
preempt_sync(struct rq *rq, int wake_flags,
	     struct sched_entity *pse, struct sched_entity *se)
{
	u64 threshold, delta;

	/*
	 * WF_SYNC without WF_TTWU is not expected so warn if it happens even
	 * though it is likely harmless.
	 */
	WARN_ON_ONCE(!(wake_flags & WF_TTWU));

	threshold = sysctl_sched_migration_cost;
	delta = rq_clock_task(rq) - se->exec_start;
	if ((s64)delta < 0)
		delta = 0;

	/*
	 * WF_RQ_SELECTED implies the tasks are stacking on a CPU when they
	 * could run on other CPUs. Reduce the threshold before preemption is
	 * allowed to an arbitrary lower value as it is more likely (but not
	 * guaranteed) the waker requires the wakee to finish.
	 */
	if (wake_flags & WF_RQ_SELECTED)
		threshold >>= 2;

	/*
	 * As WF_SYNC is not strictly obeyed, allow some runtime for batch
	 * wakeups to be issued.
	 */
	if (entity_before(pse, se) && delta >= threshold)
		return PREEMPT_WAKEUP_RESCHED;

	return PREEMPT_WAKEUP_NONE;
}

/*
 * Preempt the current task with a newly woken task if needed:
 */
/*
 * check_preempt_wakeup_fair - 判断新唤醒任务是否应促使当前任务让出 CPU。
 * 先统一组层级，再考虑空闲策略、时间片保护、buddy 和 EEVDF 选择结果。
 */
static void check_preempt_wakeup_fair(struct rq *rq, struct task_struct *p, int wake_flags)
{
	enum preempt_wakeup_action preempt_action = PREEMPT_WAKEUP_PICK;
	struct task_struct *donor = rq->donor;
	struct sched_entity *nse, *se = &donor->se, *pse = &p->se;
	struct cfs_rq *cfs_rq = task_cfs_rq(donor);
	int cse_is_idle, pse_is_idle;

	if (unlikely(se == pse))
		return;

	/*
	 * This is possible from callers such as attach_tasks(), in which we
	 * unconditionally wakeup_preempt() after an enqueue (which may have
	 * lead to a throttle).  This both saves work and prevents false
	 * next-buddy nomination below.
	 */
	if (task_is_throttled(p))
		return;

	/*
	 * We can come here with TIF_NEED_RESCHED already set from new task
	 * wake up path.
	 *
	 * Note: this also catches the edge-case of curr being in a throttled
	 * group (e.g. via set_curr_task), since update_curr() (in the
	 * enqueue of curr) will have resulted in resched being set.  This
	 * prevents us from potentially nominating it as a false LAST_BUDDY
	 * below.
	 */
	if (test_tsk_need_resched(rq->curr))
		return;

	if (!sched_feat(WAKEUP_PREEMPTION))
		return;

	/*
	 * 不同 cgroup 的叶子任务不能直接比较，
	 * 先找到在同一父队列竞争的实体。
	 */
	find_matching_se(&se, &pse);
	WARN_ON_ONCE(!pse);

	cse_is_idle = se_is_idle(se);
	pse_is_idle = se_is_idle(pse);

	/*
	 * Preempt an idle entity in favor of a non-idle entity (and don't preempt
	 * in the inverse case).
	 */
	if (cse_is_idle && !pse_is_idle) {
		/*
		 * When non-idle entity preempt an idle entity,
		 * don't give idle entity slice protection.
		 */
		preempt_action = PREEMPT_WAKEUP_SHORT;
		goto preempt;
	}

	if (cse_is_idle != pse_is_idle)
		return;

	/*
	 * BATCH and IDLE tasks do not preempt others.
	 */
	if (unlikely(!normal_policy(p->policy)))
		return;

	cfs_rq = cfs_rq_of(se);
	update_curr(cfs_rq);
	/*
	 * If @p has a shorter slice than current and @p is eligible, override
	 * current's slice protection in order to allow preemption.
	 */
	if (sched_feat(PREEMPT_SHORT) && (pse->slice < se->slice)) {
		preempt_action = PREEMPT_WAKEUP_SHORT;
		goto pick;
	}

	/*
	 * Ignore wakee preemption on WF_FORK as it is less likely that
	 * there is shared data as exec often follow fork. Do not
	 * preempt for tasks that are sched_delayed as it would violate
	 * EEVDF to forcibly queue an ineligible task.
	 */
	if ((wake_flags & WF_FORK) || pse->sched_delayed)
		return;

	/* Prefer picking wakee soon if appropriate. */
	if (sched_feat(NEXT_BUDDY) &&
	    set_preempt_buddy(cfs_rq, wake_flags, pse, se)) {

		/*
		 * Decide whether to obey WF_SYNC hint for a new buddy. Old
		 * buddies are ignored as they may not be relevant to the
		 * waker and less likely to be cache hot.
		 */
		if (wake_flags & WF_SYNC)
			preempt_action = preempt_sync(rq, wake_flags, pse, se);
	}

	switch (preempt_action) {
	case PREEMPT_WAKEUP_NONE:
		return;
	case PREEMPT_WAKEUP_RESCHED:
		goto preempt;
	case PREEMPT_WAKEUP_SHORT:
		fallthrough;
	case PREEMPT_WAKEUP_PICK:
		break;
	}

pick:
	/*
	 * 短时间片策略可绕过当前实体的保护，
	 * 但仍通过候选选取检查运行资格。
	 */
	nse = pick_next_entity(rq, cfs_rq, preempt_action != PREEMPT_WAKEUP_SHORT);
	/* If @p has become the most eligible task, force preemption */
	if (nse == pse)
		goto preempt;

	/*
	 * Because p is enqueued, nse being null can only mean that we
	 * dequeued a delayed task. If there are still entities queued in
	 * cfs, check if the next one will be p.
	 */
	if (!nse && cfs_rq->nr_queued)
		goto pick;

	if (sched_feat(RUN_TO_PARITY))
		update_protect_slice(cfs_rq, se);

	return;

preempt:
	if (preempt_action == PREEMPT_WAKEUP_SHORT)
		cancel_protect_slice(se);

	resched_curr_lazy(rq);
}

/*
 * pick_task_fair - 从根队列逐层选择 EEVDF 实体，直到落到一个任务。
 * 遇到延迟出队重新选择，遇到层级限流则为最终任务安排限流工作。
 */
static struct task_struct *pick_task_fair(struct rq *rq)
{
	struct sched_entity *se;
	struct cfs_rq *cfs_rq;
	struct task_struct *p;
	bool throttled;

again:
	cfs_rq = &rq->cfs;
	if (!cfs_rq->nr_queued)
		return NULL;

	throttled = false;

	do {
		/* Might not have done put_prev_entity() */
		if (cfs_rq->curr && cfs_rq->curr->on_rq)
			update_curr(cfs_rq);

		throttled |= check_cfs_rq_runtime(cfs_rq);

		se = pick_next_entity(rq, cfs_rq, true);
		if (!se)
			goto again;
		cfs_rq = group_cfs_rq(se);
	} while (cfs_rq);

	p = task_of(se);
	if (unlikely(throttled))
		task_throttle_setup_work(p);
	return p;
}

static void __set_next_task_fair(struct rq *rq, struct task_struct *p, bool first);
static void set_next_task_fair(struct rq *rq, struct task_struct *p, bool first);

/*
 * pick_next_task_fair - 选择下一公平任务并更新前后任务的实体状态。
 * 同组切换只处理发生变化的层级；空队列尝试拉任务，
 * 必要时要求核心重试。
 */
struct task_struct *
pick_next_task_fair(struct rq *rq, struct task_struct *prev, struct rq_flags *rf)
{
	struct sched_entity *se;
	struct task_struct *p;
	int new_tasks;

again:
	p = pick_task_fair(rq);
	if (!p)
		goto idle;
	se = &p->se;

#ifdef CONFIG_FAIR_GROUP_SCHED
	if (prev->sched_class != &fair_sched_class)
		goto simple;

	__put_prev_set_next_dl_server(rq, prev, p);

	/*
	 * Because of the set_next_buddy() in dequeue_task_fair() it is rather
	 * likely that a next task is from the same cgroup as the current.
	 *
	 * Therefore attempt to avoid putting and setting the entire cgroup
	 * hierarchy, only change the part that actually changes.
	 *
	 * Since we haven't yet done put_prev_entity and if the selected task
	 * is a different task than we started out with, try and touch the
	 * least amount of cfs_rqs.
	 */
	/*
	 * 只更新前后任务分叉的层级；共同祖先已是当前实体，
	 * 无需重复出树入树。
	 */
	if (prev != p) {
		struct sched_entity *pse = &prev->se;
		struct cfs_rq *cfs_rq;

		while (!(cfs_rq = is_same_group(se, pse))) {
			int se_depth = se->depth;
			int pse_depth = pse->depth;

			if (se_depth <= pse_depth) {
				put_prev_entity(cfs_rq_of(pse), pse);
				pse = parent_entity(pse);
			}
			if (se_depth >= pse_depth) {
				set_next_entity(cfs_rq_of(se), se, true);
				se = parent_entity(se);
			}
		}

		put_prev_entity(cfs_rq, pse);
		set_next_entity(cfs_rq, se, true);

		__set_next_task_fair(rq, p, true);
	}

	return p;

simple:
#endif /* CONFIG_FAIR_GROUP_SCHED */
	put_prev_set_next_task(rq, prev, p);
	return p;

idle:
	if (rf) {
		new_tasks = sched_balance_newidle(rq, rf);
		/* The donor may have changed while the rq lock was dropped. */
		prev = rq->donor;

		/*
		 * Because sched_balance_newidle() releases (and re-acquires)
		 * rq->lock, it is possible for any higher priority task to
		 * appear. In that case we must re-start the pick_next_entity()
		 * loop.
		 */
		/*
		 * 空闲均衡期间可能释放 rq 锁，若其他调度类已有任务，
		 * 要求核心从头重选。
		 */
		if (new_tasks < 0)
			return RETRY_TASK;

		if (new_tasks > 0)
			goto again;
	}

	return NULL;
}

/*
 * __pick_next_task_fair - 为调度类回调适配选取接口，
 * 不在此入口执行需要 rq_flags 的空闲均衡。
 */
static struct task_struct *__pick_next_task_fair(struct rq *rq, struct task_struct *prev)
{
	return pick_next_task_fair(rq, prev, NULL);
}

/*
 * fair_server_pick_task - 为截止期 fair_server 提供公平任务选择回调。
 */
static struct task_struct *fair_server_pick_task(struct sched_dl_entity *dl_se)
{
	return pick_task_fair(dl_se->rq);
}

/*
 * fair_server_init - 初始化每 CPU 的公平服务实体，并绑定公平任务选择函数。
 */
void fair_server_init(struct rq *rq)
{
	struct sched_dl_entity *dl_se = &rq->fair_server;

	init_dl_entity(dl_se);

	dl_server_init(dl_se, rq, fair_server_pick_task);
}

/*
 * Account for a descheduled task:
 */
/*
 * put_prev_task_fair - 沿组层级结算上一任务，
 * 将仍可调度的实体放回各层红黑树。
 */
static void put_prev_task_fair(struct rq *rq, struct task_struct *prev, struct task_struct *next)
{
	struct sched_entity *se = &prev->se;
	struct cfs_rq *cfs_rq;

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);
		put_prev_entity(cfs_rq, se);
	}
}

/*
 * sched_yield() is very simple
 */
/*
 * yield_task_fair - 实现公平任务主动让出 CPU：结算运行时间，
 * 并按资格放弃剩余虚拟时间片。
 */
static void yield_task_fair(struct rq *rq)
{
	struct task_struct *curr = rq->donor;
	struct cfs_rq *cfs_rq = task_cfs_rq(curr);
	struct sched_entity *se = &curr->se;

	/*
	 * Are we the only task in the tree?
	 */
	if (unlikely(rq->nr_running == 1))
		return;

	clear_buddies(cfs_rq, se);

	update_rq_clock(rq);
	/*
	 * Update run-time statistics of the 'current'.
	 */
	update_curr(cfs_rq);
	/*
	 * Tell update_rq_clock() that we've just updated,
	 * so we don't do microscopic update in schedule()
	 * and double the fastpath cost.
	 */
	rq_clock_skip_update(rq);

	/*
	 * Forfeit the remaining vruntime, only if the entity is eligible. This
	 * condition is necessary because in core scheduling we prefer to run
	 * ineligible tasks rather than force idling. If this happens we may
	 * end up in a loop where the core scheduler picks the yielding task,
	 * which yields immediately again; without the condition the vruntime
	 * ends up quickly running away.
	 */
	if (entity_eligible(cfs_rq, se)) {
		se->vruntime = se->deadline;
		update_deadline(cfs_rq, se);
	}
}

/*
 * yield_to_task_fair - 为指定在队任务设置下一候选提示，
 * 再让出当前任务的剩余竞争机会。
 */
static bool yield_to_task_fair(struct rq *rq, struct task_struct *p)
{
	struct sched_entity *se = &p->se;

	/* !se->on_rq also covers throttled task */
	if (!se->on_rq)
		return false;

	/* Tell the scheduler that we'd really like se to run next. */
	set_next_buddy(se);

	yield_task_fair(rq);

	return true;
}

/**************************************************
 * Fair scheduling class load-balancing methods.
 *
 * BASICS
 *
 * The purpose of load-balancing is to achieve the same basic fairness the
 * per-CPU scheduler provides, namely provide a proportional amount of compute
 * time to each task. This is expressed in the following equation:
 *
 *   W_i,n/P_i == W_j,n/P_j for all i,j                               (1)
 *
 * Where W_i,n is the n-th weight average for CPU i. The instantaneous weight
 * W_i,0 is defined as:
 *
 *   W_i,0 = \Sum_j w_i,j                                             (2)
 *
 * Where w_i,j is the weight of the j-th runnable task on CPU i. This weight
 * is derived from the nice value as per sched_prio_to_weight[].
 *
 * The weight average is an exponential decay average of the instantaneous
 * weight:
 *
 *   W'_i,n = (2^n - 1) / 2^n * W_i,n + 1 / 2^n * W_i,0               (3)
 *
 * C_i is the compute capacity of CPU i, typically it is the
 * fraction of 'recent' time available for SCHED_OTHER task execution. But it
 * can also include other factors [XXX].
 *
 * To achieve this balance we define a measure of imbalance which follows
 * directly from (1):
 *
 *   imb_i,j = max{ avg(W/C), W_i/C_i } - min{ avg(W/C), W_j/C_j }    (4)
 *
 * We them move tasks around to minimize the imbalance. In the continuous
 * function space it is obvious this converges, in the discrete case we get
 * a few fun cases generally called infeasible weight scenarios.
 *
 * [XXX expand on:
 *     - infeasible weights;
 *     - local vs global optima in the discrete case. ]
 *
 *
 * SCHED DOMAINS
 *
 * In order to solve the imbalance equation (4), and avoid the obvious O(n^2)
 * for all i,j solution, we create a tree of CPUs that follows the hardware
 * topology where each level pairs two lower groups (or better). This results
 * in O(log n) layers. Furthermore we reduce the number of CPUs going up the
 * tree to only the first of the previous level and we decrease the frequency
 * of load-balance at each level inversely proportional to the number of CPUs in
 * the groups.
 *
 * This yields:
 *
 *     log_2 n     1     n
 *   \Sum       { --- * --- * 2^i } = O(n)                            (5)
 *     i = 0      2^i   2^i
 *                               `- size of each group
 *         |         |     `- number of CPUs doing load-balance
 *         |         `- freq
 *         `- sum over all levels
 *
 * Coupled with a limit on how many tasks we can migrate every balance pass,
 * this makes (5) the runtime complexity of the balancer.
 *
 * An important property here is that each CPU is still (indirectly) connected
 * to every other CPU in at most O(log n) steps:
 *
 * The adjacency matrix of the resulting graph is given by:
 *
 *             log_2 n
 *   A_i,j = \Union     (i % 2^k == 0) && i / 2^(k+1) == j / 2^(k+1)  (6)
 *             k = 0
 *
 * And you'll find that:
 *
 *   A^(log_2 n)_i,j != 0  for all i,j                                (7)
 *
 * Showing there's indeed a path between every CPU in at most O(log n) steps.
 * The task movement gives a factor of O(m), giving a convergence complexity
 * of:
 *
 *   O(nm log n),  n := nr_cpus, m := nr_tasks                        (8)
 *
 *
 * WORK CONSERVING
 *
 * In order to avoid CPUs going idle while there's still work to do, new idle
 * balancing is more aggressive and has the newly idle CPU iterate up the domain
 * tree itself instead of relying on other CPUs to bring it work.
 *
 * This adds some complexity to both (5) and (8) but it reduces the total idle
 * time.
 *
 * [XXX more?]
 *
 *
 * CGROUPS
 *
 * Cgroups make a horror show out of (2), instead of a simple sum we get:
 *
 *                                s_k,i
 *   W_i,0 = \Sum_j \Prod_k w_k * -----                               (9)
 *                                 S_k
 *
 * Where
 *
 *   s_k,i = \Sum_j w_i,j,k  and  S_k = \Sum_i s_k,i                 (10)
 *
 * w_i,j,k is the weight of the j-th runnable task in the k-th cgroup on CPU i.
 *
 * The big problem is S_k, its a global sum needed to compute a local (W_i)
 * property.
 *
 * [XXX write more on how we solve this.. _after_ merging pjt's patches that
 *      rewrite all of this once again.]
 */

static unsigned long __read_mostly max_load_balance_interval = HZ/10;

enum fbq_type { regular, remote, all };

/*
 * 'group_type' describes the group of CPUs at the moment of load balancing.
 *
 * The enum is ordered by pulling priority, with the group with lowest priority
 * first so the group_type can simply be compared when selecting the busiest
 * group. See update_sd_pick_busiest().
 */
/*
 * 此处的 group 是调度拓扑中的 CPU 集合，不是 cgroup 任务组。
 * 枚举顺序体现作为迁出来源的优先级，最忙组选择可直接比较枚举大小。
 */
enum group_type {
	/* The group has spare capacity that can be used to run more tasks.  */
	group_has_spare = 0,
	/*
	 * The group is fully used and the tasks don't compete for more CPU
	 * cycles. Nevertheless, some tasks might wait before running.
	 */
	group_fully_busy,
	/*
	 * One task doesn't fit with CPU's capacity and must be migrated to a
	 * more powerful CPU.
	 */
	group_misfit_task,
	/*
	 * Balance SMT group that's fully busy. Can benefit from migration
	 * a task on SMT with busy sibling to another CPU on idle core.
	 */
	group_smt_balance,
	/*
	 * SD_ASYM_PACKING only: One local CPU with higher capacity is available,
	 * and the task should be migrated to it instead of running on the
	 * current CPU.
	 */
	group_asym_packing,
	/*
	 * The tasks' affinity constraints previously prevented the scheduler
	 * from balancing the load across the system.
	 */
	group_imbalanced,
	/*
	 * The CPU is overloaded and can't provide expected CPU cycles to all
	 * tasks.
	 */
	group_overloaded
};

/*
 * 迁移预算分别按加权负载、利用率、任务数量或容量不匹配任务计量。
 * 选源队列、计算 imbalance 和摘取任务必须使用同一种计量方式。
 */
enum migration_type {
	migrate_load = 0,
	migrate_util,
	migrate_task,
	migrate_misfit
};

#define LBF_ALL_PINNED	0x01
#define LBF_NEED_BREAK	0x02
#define LBF_DST_PINNED  0x04
#define LBF_SOME_PINNED	0x08
#define LBF_ACTIVE_LB	0x10

/*
 * 一次跨 CPU 均衡的工作上下文：
 * 记录源/目标、扫描边界、迁移预算和临时任务链表。
 * 任务先在源 rq 锁下摘取，释放源锁后再在目标 rq 锁下接入。
 */
struct lb_env {
	struct sched_domain	*sd;

	struct rq		*src_rq;
	int			src_cpu;

	int			dst_cpu;
	struct rq		*dst_rq;
	bool			dst_core_idle;

	struct cpumask		*dst_grpmask;
	int			new_dst_cpu;
	enum cpu_idle_type	idle;
	long			imbalance;
	/* The set of CPUs under consideration for load-balancing */
	struct cpumask		*cpus;

	unsigned int		flags;

	unsigned int		loop;
	unsigned int		loop_break;
	unsigned int		loop_max;

	enum fbq_type		fbq_type;
	enum migration_type	migration_type;
	struct list_head	tasks;
};

/*
 * Is this task likely cache-hot:
 */
/*
 * task_hot - 按最近执行时间、buddy 和拓扑等条件评估迁移代价，
 * 避免过早搬走热任务。
 */
static int task_hot(struct task_struct *p, struct lb_env *env)
{
	s64 delta;

	lockdep_assert_rq_held(env->src_rq);

	if (p->sched_class != &fair_sched_class)
		return 0;

	if (unlikely(task_has_idle_policy(p)))
		return 0;

	/* SMT siblings share cache */
	if (env->sd->flags & SD_SHARE_CPUCAPACITY)
		return 0;

	/*
	 * Buddy candidates are cache hot:
	 */
	if (sched_feat(CACHE_HOT_BUDDY) && env->dst_rq->nr_running &&
	    (&p->se == cfs_rq_of(&p->se)->next))
		return 1;

	if (sysctl_sched_migration_cost == -1)
		return 1;

	/*
	 * Don't migrate task if the task's cookie does not match
	 * with the destination CPU's core cookie.
	 */
	if (!sched_core_cookie_match(cpu_rq(env->dst_cpu), p))
		return 1;

	if (sysctl_sched_migration_cost == 0)
		return 0;

	delta = rq_clock_task(env->src_rq) - p->se.exec_start;

	return delta < (s64)sysctl_sched_migration_cost;
}

#ifdef CONFIG_NUMA_BALANCING
/*
 * Returns a positive value, if task migration degrades locality.
 * Returns 0, if task migration is not affected by locality.
 * Returns a negative value, if task migration improves locality i.e migration preferred.
 */
/*
 * migrate_degrades_locality - 估算迁移对 NUMA 局部性的影响：正值变差，
 * 负值改善，零表示无偏好。
 */
static long migrate_degrades_locality(struct task_struct *p, struct lb_env *env)
{
	struct numa_group *numa_group = rcu_dereference(p->numa_group);
	unsigned long src_weight, dst_weight;
	int src_nid, dst_nid, dist;

	if (!static_branch_likely(&sched_numa_balancing))
		return 0;

	if (!p->numa_faults || !(env->sd->flags & SD_NUMA))
		return 0;

	src_nid = cpu_to_node(env->src_cpu);
	dst_nid = cpu_to_node(env->dst_cpu);

	if (src_nid == dst_nid)
		return 0;

	/* Migrating away from the preferred node is always bad. */
	if (src_nid == p->numa_preferred_nid) {
		if (env->src_rq->nr_running > env->src_rq->nr_preferred_running)
			return 1;
		else
			return 0;
	}

	/* Encourage migration to the preferred node. */
	if (dst_nid == p->numa_preferred_nid)
		return -1;

	/* Leaving a core idle is often worse than degrading locality. */
	if (env->idle == CPU_IDLE)
		return 0;

	dist = node_distance(src_nid, dst_nid);
	if (numa_group) {
		src_weight = group_weight(p, src_nid, dist);
		dst_weight = group_weight(p, dst_nid, dist);
	} else {
		src_weight = task_weight(p, src_nid, dist);
		dst_weight = task_weight(p, dst_nid, dist);
	}

	return src_weight - dst_weight;
}

#else /* !CONFIG_NUMA_BALANCING: */
/*
 * migrate_degrades_locality - 未启用 NUMA 自动均衡时，
 * 不对局部性变化作额外评分。
 */
static inline long migrate_degrades_locality(struct task_struct *p,
					     struct lb_env *env)
{
	return 0;
}
#endif /* !CONFIG_NUMA_BALANCING */

/*
 * Check whether the task is ineligible on the destination cpu
 *
 * When the PLACE_LAG scheduling feature is enabled and
 * dst_cfs_rq->nr_queued is greater than 1, if the task
 * is ineligible, it will also be ineligible when
 * it is migrated to the destination cpu.
 */
/*
 * task_is_ineligible_on_dst_cpu - 若目标非空且放置策略保留 lag，
 * 检查任务迁入后是否仍无运行资格。
 */
static inline int task_is_ineligible_on_dst_cpu(struct task_struct *p, int dest_cpu)
{
	struct cfs_rq *dst_cfs_rq;

#ifdef CONFIG_FAIR_GROUP_SCHED
	dst_cfs_rq = task_group(p)->cfs_rq[dest_cpu];
#else
	dst_cfs_rq = &cpu_rq(dest_cpu)->cfs;
#endif
	if (sched_feat(PLACE_LAG) && dst_cfs_rq->nr_queued &&
	    !entity_eligible(task_cfs_rq(p), &p->se))
		return 1;

	return 0;
}

/*
 * can_migrate_task - may task p from runqueue rq be migrated to this_cpu?
 */
/*
 * can_migrate_task - 筛选可迁移任务，
 * 检查限流、亲和性、执行状态、运行资格及迁移代价。
 * 主动均衡或多次失败可放宽热度限制，但不能绕过亲和性等硬约束。
 */
static
int can_migrate_task(struct task_struct *p, struct lb_env *env)
{
	long degrades, hot;

	lockdep_assert_rq_held(env->src_rq);
	if (p->sched_task_hot)
		p->sched_task_hot = 0;

	/*
	 * We do not migrate tasks that are:
	 * 1) delayed dequeued unless we migrate load, or
	 * 2) target cfs_rq is in throttled hierarchy, or
	 * 3) cannot be migrated to this CPU due to cpus_ptr, or
	 * 4) running (obviously), or
	 * 5) are cache-hot on their current CPU, or
	 * 6) are blocked on mutexes (if SCHED_PROXY_EXEC is enabled)
	 */
	if ((p->se.sched_delayed) && (env->migration_type != migrate_load))
		return 0;

	if (lb_throttled_hierarchy(p, env->dst_cpu))
		return 0;

	/*
	 * We want to prioritize the migration of eligible tasks.
	 * For ineligible tasks we soft-limit them and only allow
	 * them to migrate when nr_balance_failed is non-zero to
	 * avoid load-balancing trying very hard to balance the load.
	 */
	if (!env->sd->nr_balance_failed &&
	    task_is_ineligible_on_dst_cpu(p, env->dst_cpu))
		return 0;

	/* Disregard percpu kthreads; they are where they need to be. */
	if (kthread_is_per_cpu(p))
		return 0;

	if (task_is_blocked(p))
		return 0;

	if (!cpumask_test_cpu(env->dst_cpu, p->cpus_ptr)) {
		int cpu;

		schedstat_inc(p->stats.nr_failed_migrations_affine);

		env->flags |= LBF_SOME_PINNED;

		/*
		 * Remember if this task can be migrated to any other CPU in
		 * our sched_group. We may want to revisit it if we couldn't
		 * meet load balance goals by pulling other tasks on src_cpu.
		 *
		 * Avoid computing new_dst_cpu
		 * - for NEWLY_IDLE
		 * - if we have already computed one in current iteration
		 * - if it's an active balance
		 */
		if (env->idle == CPU_NEWLY_IDLE ||
		    env->flags & (LBF_DST_PINNED | LBF_ACTIVE_LB))
			return 0;

		/* Prevent to re-select dst_cpu via env's CPUs: */
		cpu = cpumask_first_and_and(env->dst_grpmask, env->cpus, p->cpus_ptr);

		if (cpu < nr_cpu_ids) {
			env->flags |= LBF_DST_PINNED;
			env->new_dst_cpu = cpu;
		}

		return 0;
	}

	/* Record that we found at least one task that could run on dst_cpu */
	env->flags &= ~LBF_ALL_PINNED;

	/*
	 * 实际执行者和代理执行的资源提供者都不能由普通拉取直接迁走。
	 */
	if (task_on_cpu(env->src_rq, p) ||
	    task_current_donor(env->src_rq, p)) {
		schedstat_inc(p->stats.nr_failed_migrations_running);
		return 0;
	}

	/*
	 * Aggressive migration if:
	 * 1) active balance
	 * 2) destination numa is preferred
	 * 3) task is cache cold, or
	 * 4) too many balance attempts have failed.
	 */
	if (env->flags & LBF_ACTIVE_LB)
		return 1;

	degrades = migrate_degrades_locality(p, env);
	if (!degrades)
		hot = task_hot(p, env);
	else
		hot = degrades > 0;

	if (!hot || env->sd->nr_balance_failed > env->sd->cache_nice_tries) {
		if (hot)
			p->sched_task_hot = 1;
		return 1;
	}

	schedstat_inc(p->stats.nr_failed_migrations_hot);
	return 0;
}

/*
 * detach_task() -- detach the task for the migration specified in env
 */
/*
 * detach_task - 持有源队列锁时停用任务并修改目标 CPU，
 * 等待后续在目标队列激活。
 */
static void detach_task(struct task_struct *p, struct lb_env *env)
{
	lockdep_assert_rq_held(env->src_rq);

	if (p->sched_task_hot) {
		p->sched_task_hot = 0;
		schedstat_inc(env->sd->lb_hot_gained[env->idle]);
		schedstat_inc(p->stats.nr_forced_migrations);
	}

	WARN_ON(task_current(env->src_rq, p));
	WARN_ON(task_current_donor(env->src_rq, p));

	deactivate_task(env->src_rq, p, DEQUEUE_NOCLOCK);
	set_task_cpu(p, env->dst_cpu);
}

/*
 * detach_one_task() -- tries to dequeue exactly one task from env->src_rq, as
 * part of active balancing operations within "domain".
 *
 * Returns a task if successful and NULL otherwise.
 */
/*
 * detach_one_task - 从源队列较久未运行的任务开始，
 * 摘取一个满足迁移条件的任务。
 */
static struct task_struct *detach_one_task(struct lb_env *env)
{
	struct task_struct *p;

	lockdep_assert_rq_held(env->src_rq);

	list_for_each_entry_reverse(p,
			&env->src_rq->cfs_tasks, se.group_node) {
		if (!can_migrate_task(p, env))
			continue;

		detach_task(p, env);

		/*
		 * Right now, this is only the second place where
		 * lb_gained[env->idle] is updated (other is detach_tasks)
		 * so we can safely collect stats here rather than
		 * inside detach_tasks().
		 */
		schedstat_inc(env->sd->lb_gained[env->idle]);
		return p;
	}
	return NULL;
}

/*
 * detach_tasks() -- tries to detach up to imbalance load/util/tasks from
 * busiest_rq, as part of a balancing operation within domain "sd".
 *
 * Returns number of detached tasks if successful and 0 otherwise.
 */
/*
 * detach_tasks - 按本轮均衡目标批量摘取任务，暂存到 env->tasks。
 * 迁移预算可按负载、利用率、任务数或容量不匹配计算，
 * 并限制持锁扫描长度。
 */
static int detach_tasks(struct lb_env *env)
{
	struct list_head *tasks = &env->src_rq->cfs_tasks;
	unsigned long util, load;
	struct task_struct *p;
	int detached = 0;

	lockdep_assert_rq_held(env->src_rq);

	/*
	 * Source run queue has been emptied by another CPU, clear
	 * LBF_ALL_PINNED flag as we will not test any task.
	 */
	if (env->src_rq->nr_running <= 1) {
		env->flags &= ~LBF_ALL_PINNED;
		return 0;
	}

	if (env->imbalance <= 0)
		return 0;

	while (!list_empty(tasks)) {
		/*
		 * We don't want to steal all, otherwise we may be treated likewise,
		 * which could at worst lead to a livelock crash.
		 */
		if (env->idle && env->src_rq->nr_running <= 1)
			break;

		env->loop++;
		/* We've more or less seen every task there is, call it quits */
		if (env->loop > env->loop_max)
			break;

		/* take a breather every nr_migrate tasks */
		/*
		 * 扫描分批让出锁，避免大队列导致其他调度操作长时间等待。
		 */
		if (env->loop > env->loop_break) {
			env->loop_break += SCHED_NR_MIGRATE_BREAK;
			env->flags |= LBF_NEED_BREAK;
			break;
		}

		p = list_last_entry(tasks, struct task_struct, se.group_node);

		if (!can_migrate_task(p, env))
			goto next;

		/*
		 * 只从当前模式对应的预算中扣除任务贡献，
		 * 不能混用权重与利用率单位。
		 */
		switch (env->migration_type) {
		case migrate_load:
			/*
			 * Depending of the number of CPUs and tasks and the
			 * cgroup hierarchy, task_h_load() can return a null
			 * value. Make sure that env->imbalance decreases
			 * otherwise detach_tasks() will stop only after
			 * detaching up to loop_max tasks.
			 */
			load = max_t(unsigned long, task_h_load(p), 1);

			if (sched_feat(LB_MIN) &&
			    load < 16 && !env->sd->nr_balance_failed)
				goto next;

			/*
			 * Make sure that we don't migrate too much load.
			 * Nevertheless, let relax the constraint if
			 * scheduler fails to find a good waiting task to
			 * migrate.
			 */
			if (shr_bound(load, env->sd->nr_balance_failed) > env->imbalance)
				goto next;

			env->imbalance -= load;
			break;

		case migrate_util:
			util = task_util_est(p);

			if (shr_bound(util, env->sd->nr_balance_failed) > env->imbalance)
				goto next;

			env->imbalance -= util;
			break;

		case migrate_task:
			env->imbalance--;
			break;

		case migrate_misfit:
			/* This is not a misfit task */
			if (task_fits_cpu(p, env->src_cpu))
				goto next;

			env->imbalance = 0;
			break;
		}

		detach_task(p, env);
		list_add(&p->se.group_node, &env->tasks);

		detached++;

#ifdef CONFIG_PREEMPTION
		/*
		 * NEWIDLE balancing is a source of latency, so preemptible
		 * kernels will stop after the first task is detached to minimize
		 * the critical section.
		 */
		if (env->idle == CPU_NEWLY_IDLE)
			break;
#endif

		/*
		 * We only want to steal up to the prescribed amount of
		 * load/util/tasks.
		 */
		if (env->imbalance <= 0)
			break;

		continue;
next:
		if (p->sched_task_hot)
			schedstat_inc(p->stats.nr_failed_migrations_hot);

		list_move(&p->se.group_node, tasks);
	}

	/*
	 * Right now, this is one of only two places we collect this stat
	 * so we can safely collect detach_one_task() stats here rather
	 * than inside detach_one_task().
	 */
	schedstat_add(env->sd->lb_gained[env->idle], detached);

	return detached;
}

/*
 * attach_task() -- attach the task detached by detach_task() to its new rq.
 */
/*
 * attach_task - 持有目标队列锁时激活迁入任务，
 * 并检查它是否应抢占当前任务。
 */
static void attach_task(struct rq *rq, struct task_struct *p)
{
	lockdep_assert_rq_held(rq);

	WARN_ON_ONCE(task_rq(p) != rq);
	activate_task(rq, p, ENQUEUE_NOCLOCK);
	wakeup_preempt(rq, p, 0);
}

/*
 * attach_one_task() -- attaches the task returned from detach_one_task() to
 * its new rq.
 */
/*
 * attach_one_task - 加锁并更新目标时钟后，完成单个任务的迁移入队。
 */
static void attach_one_task(struct rq *rq, struct task_struct *p)
{
	struct rq_flags rf;

	rq_lock(rq, &rf);
	update_rq_clock(rq);
	attach_task(rq, p);
	rq_unlock(rq, &rf);
}

/*
 * attach_tasks() -- attaches all tasks detached by detach_tasks() to their
 * new rq.
 */
/*
 * attach_tasks - 批量把临时迁移链表中的任务接入目标队列。
 */
static void attach_tasks(struct lb_env *env)
{
	struct list_head *tasks = &env->tasks;
	struct task_struct *p;
	struct rq_flags rf;

	rq_lock(env->dst_rq, &rf);
	update_rq_clock(env->dst_rq);

	while (!list_empty(tasks)) {
		p = list_first_entry(tasks, struct task_struct, se.group_node);
		list_del_init(&p->se.group_node);

		attach_task(env->dst_rq, p);
	}

	rq_unlock(env->dst_rq, &rf);
}

#ifdef CONFIG_NO_HZ_COMMON
/*
 * cfs_rq_has_blocked：
 * 判断公平队列是否仍有非零历史负载或利用率需要继续衰减。
 */
static inline bool cfs_rq_has_blocked(struct cfs_rq *cfs_rq)
{
	if (cfs_rq->avg.load_avg)
		return true;

	if (cfs_rq->avg.util_avg)
		return true;

	return false;
}

/*
 * others_have_blocked：
 * 检查实时、截止期、硬件或中断的历史负载是否仍需更新。
 */
static inline bool others_have_blocked(struct rq *rq)
{
	if (cpu_util_rt(rq))
		return true;

	if (cpu_util_dl(rq))
		return true;

	if (hw_load_avg(rq))
		return true;

	if (cpu_util_irq(rq))
		return true;

	return false;
}

/*
 * update_blocked_load_tick - 记录最近一次阻塞负载维护的 jiffies，
 * 避免同一 tick 重复更新。
 */
static inline void update_blocked_load_tick(struct rq *rq)
{
	WRITE_ONCE(rq->last_blocked_load_update_tick, jiffies);
}

/*
 * update_blocked_load_status - 所有历史贡献均衰减完时清除待维护标志。
 */
static inline void update_blocked_load_status(struct rq *rq, bool has_blocked)
{
	if (!has_blocked)
		rq->has_blocked_load = 0;
}
#else /* !CONFIG_NO_HZ_COMMON: */
/*
 * cfs_rq_has_blocked - 未启用 NO_HZ 公共支持时，
 * 不维护供停 tick 路径使用的阻塞标志。
 */
static inline bool cfs_rq_has_blocked(struct cfs_rq *cfs_rq) { return false; }
/*
 * others_have_blocked - 未启用 NO_HZ 公共支持时，
 * 不汇总其他负载的停 tick 维护状态。
 */
static inline bool others_have_blocked(struct rq *rq) { return false; }
/*
 * update_blocked_load_tick - 未启用 NO_HZ 公共支持时，无需保存停 tick 维护时间。
 */
static inline void update_blocked_load_tick(struct rq *rq) {}
/*
 * update_blocked_load_status - 未启用 NO_HZ 公共支持时，
 * 无需更新停 tick 维护标志。
 */
static inline void update_blocked_load_status(struct rq *rq, bool has_blocked) {}
#endif /* !CONFIG_NO_HZ_COMMON */

/*
 * __update_blocked_others - 衰减其他调度类及中断等负载，
 * 并反馈是否仍有残留贡献。
 */
static bool __update_blocked_others(struct rq *rq, bool *done)
{
	bool updated;

	/*
	 * update_load_avg() can call cpufreq_update_util(). Make sure that RT,
	 * DL and IRQ signals have been updated before updating CFS.
	 */
	updated = update_other_load_avgs(rq);

	if (others_have_blocked(rq))
		*done = false;

	return updated;
}

#ifdef CONFIG_FAIR_GROUP_SCHED

/*
 * __update_blocked_fair - 按子先父后的顺序衰减组队列负载，
 * 将变化传播到根并移除已清空队列。
 */
static bool __update_blocked_fair(struct rq *rq, bool *done)
{
	struct cfs_rq *cfs_rq, *pos;
	bool decayed = false;
	int cpu = cpu_of(rq);

	/*
	 * Iterates the task_group tree in a bottom up fashion, see
	 * list_add_leaf_cfs_rq() for details.
	 */
	for_each_leaf_cfs_rq_safe(rq, cfs_rq, pos) {
		struct sched_entity *se;

		if (update_cfs_rq_load_avg(cfs_rq_clock_pelt(cfs_rq), cfs_rq)) {
			update_tg_load_avg(cfs_rq);

			if (cfs_rq->nr_queued == 0)
				update_idle_cfs_rq_clock_pelt(cfs_rq);

			if (cfs_rq == &rq->cfs)
				decayed = true;
		}

		/* Propagate pending load changes to the parent, if any: */
		se = cfs_rq->tg->se[cpu];
		if (se && !skip_blocked_update(se))
			update_load_avg(cfs_rq_of(se), se, UPDATE_TG);

		/*
		 * There can be a lot of idle CPU cgroups.  Don't let fully
		 * decayed cfs_rqs linger on the list.
		 */
		if (cfs_rq_is_decayed(cfs_rq))
			list_del_leaf_cfs_rq(cfs_rq);

		/* Don't need periodic decay once load/util_avg are null */
		if (cfs_rq_has_blocked(cfs_rq))
			*done = false;
	}

	return decayed;
}

/*
 * Compute the hierarchical load factor for cfs_rq and all its ascendants.
 * This needs to be done in a top-down fashion because the load of a child
 * group is a fraction of its parents load.
 */
/*
 * update_cfs_rq_h_load - 沿组层级计算队列折算到根队列的负载份额，
 * 同一 jiffy 内复用结果。
 */
static void update_cfs_rq_h_load(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	struct sched_entity *se = cfs_rq->tg->se[cpu_of(rq)];
	unsigned long now = jiffies;
	unsigned long load;

	if (cfs_rq->last_h_load_update == now)
		return;

	WRITE_ONCE(cfs_rq->h_load_next, NULL);
	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);
		WRITE_ONCE(cfs_rq->h_load_next, se);
		if (cfs_rq->last_h_load_update == now)
			break;
	}

	if (!se) {
		cfs_rq->h_load = cfs_rq_load_avg(cfs_rq);
		cfs_rq->last_h_load_update = now;
	}

	while ((se = READ_ONCE(cfs_rq->h_load_next)) != NULL) {
		load = cfs_rq->h_load;
		load = div64_ul(load * se->avg.load_avg,
			cfs_rq_load_avg(cfs_rq) + 1);
		cfs_rq = group_cfs_rq(se);
		cfs_rq->h_load = load;
		cfs_rq->last_h_load_update = now;
	}
}

/*
 * task_h_load - 把任务自身负载按各级组份额折算为 CPU 根队列尺度，
 * 供均衡比较。
 */
static unsigned long task_h_load(struct task_struct *p)
{
	struct cfs_rq *cfs_rq = task_cfs_rq(p);

	update_cfs_rq_h_load(cfs_rq);
	return div64_ul(p->se.avg.load_avg * cfs_rq->h_load,
			cfs_rq_load_avg(cfs_rq) + 1);
}
#else /* !CONFIG_FAIR_GROUP_SCHED: */
/*
 * __update_blocked_fair - 未启用组调度时，直接更新根公平队列的历史负载。
 */
static bool __update_blocked_fair(struct rq *rq, bool *done)
{
	struct cfs_rq *cfs_rq = &rq->cfs;
	bool decayed;

	decayed = update_cfs_rq_load_avg(cfs_rq_clock_pelt(cfs_rq), cfs_rq);
	if (cfs_rq_has_blocked(cfs_rq))
		*done = false;

	return decayed;
}

/*
 * task_h_load - 未启用组调度时，任务自身 PELT 负载即为均衡使用的负载。
 */
static unsigned long task_h_load(struct task_struct *p)
{
	return p->se.avg.load_avg;
}
#endif /* !CONFIG_FAIR_GROUP_SCHED */

/*
 * sched_balance_update_blocked_averages - 加锁推进指定 CPU 的各类历史负载，
 * 并按需通知调频。
 */
static void sched_balance_update_blocked_averages(int cpu)
{
	bool decayed = false, done = true;
	struct rq *rq = cpu_rq(cpu);
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);
	update_blocked_load_tick(rq);
	update_rq_clock(rq);

	decayed |= __update_blocked_others(rq, &done);
	decayed |= __update_blocked_fair(rq, &done);

	update_blocked_load_status(rq, !done);
	if (decayed)
		cpufreq_update_util(rq, 0);
	rq_unlock_irqrestore(rq, &rf);
}

/********** Helpers for sched_balance_find_src_group ************************/

/*
 * sg_lb_stats - stats of a sched_group required for load-balancing:
 */
struct sg_lb_stats {
	unsigned long avg_load;			/* Avg load            over the CPUs of the group */
	unsigned long group_load;		/* Total load          over the CPUs of the group */
	unsigned long group_capacity;		/* Capacity            over the CPUs of the group */
	unsigned long group_util;		/* Total utilization   over the CPUs of the group */
	unsigned long group_runnable;		/* Total runnable time over the CPUs of the group */
	unsigned int sum_nr_running;		/* Nr of all tasks running in the group */
	unsigned int sum_h_nr_running;		/* Nr of CFS tasks running in the group */
	unsigned int idle_cpus;                 /* Nr of idle CPUs         in the group */
	unsigned int group_weight;
	enum group_type group_type;
	unsigned int group_asym_packing;	/* Tasks should be moved to preferred CPU */
	unsigned int group_smt_balance;		/* Task on busy SMT be moved */
	unsigned long group_misfit_task_load;	/* A CPU has a task too big for its capacity */
#ifdef CONFIG_NUMA_BALANCING
	unsigned int nr_numa_running;
	unsigned int nr_preferred_running;
#endif
};

/*
 * sd_lb_stats - stats of a sched_domain required for load-balancing:
 */
struct sd_lb_stats {
	struct sched_group *busiest;		/* Busiest group in this sd */
	struct sched_group *local;		/* Local group in this sd */
	unsigned long total_load;		/* Total load of all groups in sd */
	unsigned long total_capacity;		/* Total capacity of all groups in sd */
	unsigned long avg_load;			/* Average load across all groups in sd */
	unsigned int prefer_sibling;		/* Tasks should go to sibling first */

	struct sg_lb_stats busiest_stat;	/* Statistics of the busiest group */
	struct sg_lb_stats local_stat;		/* Statistics of the local group */
};

/*
 * init_sd_lb_stats - 初始化本轮调度域统计和最忙组比较的基准值。
 */
static inline void init_sd_lb_stats(struct sd_lb_stats *sds)
{
	/*
	 * Skimp on the clearing to avoid duplicate work. We can avoid clearing
	 * local_stat because update_sg_lb_stats() does a full clear/assignment.
	 * We must however set busiest_stat::group_type and
	 * busiest_stat::idle_cpus to the worst busiest group because
	 * update_sd_pick_busiest() reads these before assignment.
	 */
	*sds = (struct sd_lb_stats){
		.busiest = NULL,
		.local = NULL,
		.total_load = 0UL,
		.total_capacity = 0UL,
		.busiest_stat = {
			.idle_cpus = UINT_MAX,
			.group_type = group_has_spare,
		},
	};
}

/*
 * scale_rt_capacity - 从实际容量中扣除实时、截止期和中断占用，
 * 估算留给公平任务的容量。
 */
static unsigned long scale_rt_capacity(int cpu)
{
	unsigned long max = get_actual_cpu_capacity(cpu);
	struct rq *rq = cpu_rq(cpu);
	unsigned long used, free;
	unsigned long irq;

	irq = cpu_util_irq(rq);

	if (unlikely(irq >= max))
		return 1;

	/*
	 * avg_rt.util_avg and avg_dl.util_avg track binary signals
	 * (running and not running) with weights 0 and 1024 respectively.
	 */
	used = cpu_util_rt(rq);
	used += cpu_util_dl(rq);

	if (unlikely(used >= max))
		return 1;

	free = max - used;

	return scale_irq_capacity(free, irq, max);
}

/*
 * update_cpu_capacity - 更新单 CPU 可用容量，并将它写入底层调度组容量统计。
 */
static void update_cpu_capacity(struct sched_domain *sd, int cpu)
{
	unsigned long capacity = scale_rt_capacity(cpu);
	struct sched_group *sdg = sd->groups;

	if (!capacity)
		capacity = 1;

	cpu_rq(cpu)->cpu_capacity = capacity;
	trace_sched_cpu_capacity_tp(cpu_rq(cpu));

	sdg->sgc->capacity = capacity;
	sdg->sgc->min_capacity = capacity;
	sdg->sgc->max_capacity = capacity;
}

/*
 * update_group_capacity - 自底向上汇总调度组总容量及最小、最大单 CPU 容量。
 */
void update_group_capacity(struct sched_domain *sd, int cpu)
{
	struct sched_domain *child = sd->child;
	struct sched_group *group, *sdg = sd->groups;
	unsigned long capacity, min_capacity, max_capacity;
	unsigned long interval;

	interval = msecs_to_jiffies(sd->balance_interval);
	interval = clamp(interval, 1UL, max_load_balance_interval);
	sdg->sgc->next_update = jiffies + interval;

	if (!child) {
		update_cpu_capacity(sd, cpu);
		return;
	}

	capacity = 0;
	min_capacity = ULONG_MAX;
	max_capacity = 0;

	if (child->flags & SD_NUMA) {
		/*
		 * SD_NUMA domains cannot assume that child groups
		 * span the current group.
		 */

		for_each_cpu(cpu, sched_group_span(sdg)) {
			unsigned long cpu_cap = capacity_of(cpu);

			capacity += cpu_cap;
			min_capacity = min(cpu_cap, min_capacity);
			max_capacity = max(cpu_cap, max_capacity);
		}
	} else  {
		/*
		 * !SD_NUMA domains can assume that child groups
		 * span the current group.
		 */

		group = child->groups;
		do {
			struct sched_group_capacity *sgc = group->sgc;

			capacity += sgc->capacity;
			min_capacity = min(sgc->min_capacity, min_capacity);
			max_capacity = max(sgc->max_capacity, max_capacity);
			group = group->next;
		} while (group != child->groups);
	}

	sdg->sgc->capacity = capacity;
	sdg->sgc->min_capacity = min_capacity;
	sdg->sgc->max_capacity = max_capacity;
}

/*
 * Check whether the capacity of the rq has been noticeably reduced by side
 * activity. The imbalance_pct is used for the threshold.
 * Return true is the capacity is reduced
 */
/*
 * check_cpu_capacity：
 * 判断公平任务可用容量是否相对架构原始容量下降到显著程度。
 */
static inline int
check_cpu_capacity(struct rq *rq, struct sched_domain *sd)
{
	return ((rq->cpu_capacity * sd->imbalance_pct) <
				(arch_scale_cpu_capacity(cpu_of(rq)) * 100));
}

/* Check if the rq has a misfit task */
/*
 * check_misfit_status - 检查运行队列是否有已标记的容量不匹配任务。
 */
static inline bool check_misfit_status(struct rq *rq)
{
	return rq->misfit_task_load;
}

/*
 * Group imbalance indicates (and tries to solve) the problem where balancing
 * groups is inadequate due to ->cpus_ptr constraints.
 *
 * Imagine a situation of two groups of 4 CPUs each and 4 tasks each with a
 * cpumask covering 1 CPU of the first group and 3 CPUs of the second group.
 * Something like:
 *
 *	{ 0 1 2 3 } { 4 5 6 7 }
 *	        *     * * *
 *
 * If we were to balance group-wise we'd place two tasks in the first group and
 * two tasks in the second group. Clearly this is undesired as it will overload
 * cpu 3 and leave one of the CPUs in the second group unused.
 *
 * The current solution to this issue is detecting the skew in the first group
 * by noticing the lower domain failed to reach balance and had difficulty
 * moving tasks due to affinity constraints.
 *
 * When this is so detected; this group becomes a candidate for busiest; see
 * update_sd_pick_busiest(). And calculate_imbalance() and
 * sched_balance_find_src_group() avoid some of the usual balance conditions to allow it
 * to create an effective group imbalance.
 *
 * This is a somewhat tricky proposition since the next run might not find the
 * group imbalance and decide the groups need to be balanced again. A most
 * subtle and fragile situation.
 */

/*
 * sg_imbalanced - 读取上轮因亲和性限制而留下的组内不均衡标志。
 */
static inline int sg_imbalanced(struct sched_group *group)
{
	return group->sgc->imbalance;
}

/*
 * group_has_capacity returns true if the group has spare capacity that could
 * be used by some tasks.
 * We consider that a group has spare capacity if the number of task is
 * smaller than the number of CPUs or if the utilization is lower than the
 * available capacity for CFS tasks.
 * For the latter, we use a threshold to stabilize the state, to take into
 * account the variance of the tasks' load and to return true if the available
 * capacity in meaningful for the load balancer.
 * As an example, an available capacity of 1% can appear but it doesn't make
 * any benefit for the load balance.
 */
/*
 * group_has_capacity - 结合任务数、需求和利用率，
 * 判断组内是否还有接纳任务的空间。
 */
static inline bool
group_has_capacity(unsigned int imbalance_pct, struct sg_lb_stats *sgs)
{
	if (sgs->sum_nr_running < sgs->group_weight)
		return true;

	if ((sgs->group_capacity * imbalance_pct) <
			(sgs->group_runnable * 100))
		return false;

	if ((sgs->group_capacity * 100) >
			(sgs->group_util * imbalance_pct))
		return true;

	return false;
}

/*
 *  group_is_overloaded returns true if the group has more tasks than it can
 *  handle.
 *  group_is_overloaded is not equals to !group_has_capacity because a group
 *  with the exact right number of tasks, has no more spare capacity but is not
 *  overloaded so both group_has_capacity and group_is_overloaded return
 *  false.
 */
/*
 * group_is_overloaded：
 * 判断任务竞争和利用率/需求是否已超过该组的容量容差。
 */
static inline bool
group_is_overloaded(unsigned int imbalance_pct, struct sg_lb_stats *sgs)
{
	if (sgs->sum_nr_running <= sgs->group_weight)
		return false;

	if ((sgs->group_capacity * 100) <
			(sgs->group_util * imbalance_pct))
		return true;

	if ((sgs->group_capacity * imbalance_pct) <
			(sgs->group_runnable * 100))
		return true;

	return false;
}

/*
 * group_classify：
 * 按过载、亲和性失衡、非对称打包等条件确定组的拉取优先级。
 */
static inline enum
group_type group_classify(unsigned int imbalance_pct,
			  struct sched_group *group,
			  struct sg_lb_stats *sgs)
{
	if (group_is_overloaded(imbalance_pct, sgs))
		return group_overloaded;

	if (sg_imbalanced(group))
		return group_imbalanced;

	if (sgs->group_asym_packing)
		return group_asym_packing;

	if (sgs->group_smt_balance)
		return group_smt_balance;

	if (sgs->group_misfit_task_load)
		return group_misfit_task;

	if (!group_has_capacity(imbalance_pct, sgs))
		return group_fully_busy;

	return group_has_spare;
}

/*
 * sched_use_asym_prio：
 * 检查当前拓扑和 SMT 空闲状态是否允许使用非对称 CPU 优先级。
 */
/**
 * sched_use_asym_prio - Check whether asym_packing priority must be used
 * @sd:		The scheduling domain of the load balancing
 * @cpu:	A CPU
 *
 * Always use CPU priority when balancing load between SMT siblings. When
 * balancing load between cores, it is not sufficient that @cpu is idle. Only
 * use CPU priority if the whole core is idle.
 *
 * Returns: True if the priority of @cpu must be followed. False otherwise.
 */
static bool sched_use_asym_prio(struct sched_domain *sd, int cpu)
{
	if (!(sd->flags & SD_ASYM_PACKING))
		return false;

	if (!sched_smt_active())
		return true;

	return sd->flags & SD_SHARE_CPUCAPACITY || is_core_idle(cpu);
}

/*
 * sched_asym - 判断目标 CPU 是否可凭非对称优先级从源 CPU 拉取任务。
 */
static inline bool sched_asym(struct sched_domain *sd, int dst_cpu, int src_cpu)
{
	/*
	 * First check if @dst_cpu can do asym_packing load balance. Only do it
	 * if it has higher priority than @src_cpu.
	 */
	return sched_use_asym_prio(sd, dst_cpu) &&
		sched_asym_prefer(dst_cpu, src_cpu);
}

/*
 * sched_group_asym - 结合源组 SMT 占用情况，
 * 判断是否应向更优先的目标 CPU 打包任务。
 */
/**
 * sched_group_asym - Check if the destination CPU can do asym_packing balance
 * @env:	The load balancing environment
 * @sgs:	Load-balancing statistics of the candidate busiest group
 * @group:	The candidate busiest group
 *
 * @env::dst_cpu can do asym_packing if it has higher priority than the
 * preferred CPU of @group.
 *
 * Return: true if @env::dst_cpu can do with asym_packing load balance. False
 * otherwise.
 */
static inline bool
sched_group_asym(struct lb_env *env, struct sg_lb_stats *sgs, struct sched_group *group)
{
	/*
	 * CPU priorities do not make sense for SMT cores with more than one
	 * busy sibling.
	 */
	if ((group->flags & SD_SHARE_CPUCAPACITY) &&
	    (sgs->group_weight - sgs->idle_cpus != 1))
		return false;

	return sched_asym(env->sd, env->dst_cpu, READ_ONCE(group->asym_prefer_cpu));
}

/* One group has more than one SMT CPU while the other group does not */
/*
 * smt_vs_nonsmt_groups：
 * 判断两个候选组是否分别属于 SMT 共享容量组与非 SMT 组。
 */
static inline bool smt_vs_nonsmt_groups(struct sched_group *sg1,
				    struct sched_group *sg2)
{
	if (!sg1 || !sg2)
		return false;

	return (sg1->flags & SD_SHARE_CPUCAPACITY) !=
		(sg2->flags & SD_SHARE_CPUCAPACITY);
}

/*
 * smt_balance - 目标空闲且源 SMT 组有多个公平任务时，考虑分散到独立核心。
 */
static inline bool smt_balance(struct lb_env *env, struct sg_lb_stats *sgs,
			       struct sched_group *group)
{
	if (!env->idle)
		return false;

	/*
	 * For SMT source group, it is better to move a task
	 * to a CPU that doesn't have multiple tasks sharing its CPU capacity.
	 * Note that if a group has a single SMT, SD_SHARE_CPUCAPACITY
	 * will not be on.
	 */
	if (group->flags & SD_SHARE_CPUCAPACITY &&
	    sgs->sum_h_nr_running > 1)
		return true;

	return false;
}

/*
 * sibling_imbalance - 按两组物理核心数折算任务分布差异，
 * 供后续决定应迁移的任务数。
 */
static inline long sibling_imbalance(struct lb_env *env,
				    struct sd_lb_stats *sds,
				    struct sg_lb_stats *busiest,
				    struct sg_lb_stats *local)
{
	int ncores_busiest, ncores_local;
	long imbalance;

	if (!env->idle || !busiest->sum_nr_running)
		return 0;

	ncores_busiest = sds->busiest->cores;
	ncores_local = sds->local->cores;

	if (ncores_busiest == ncores_local) {
		imbalance = busiest->sum_nr_running;
		lsub_positive(&imbalance, local->sum_nr_running);
		return imbalance;
	}

	/* Balance such that nr_running/ncores ratio are same on both groups */
	imbalance = ncores_local * busiest->sum_nr_running;
	lsub_positive(&imbalance, ncores_busiest * local->sum_nr_running);
	/* Normalize imbalance and do rounding on normalization */
	imbalance = 2 * imbalance + ncores_local + ncores_busiest;
	imbalance /= ncores_local + ncores_busiest;

	/* Take advantage of resource in an empty sched group */
	if (imbalance <= 1 && local->sum_nr_running == 0 &&
	    busiest->sum_nr_running > 1)
		imbalance = 2;

	return imbalance;
}

/*
 * sched_reduced_capacity：
 * 识别只有一个公平任务但可用容量已明显降低的运行队列。
 */
static inline bool
sched_reduced_capacity(struct rq *rq, struct sched_domain *sd)
{
	/*
	 * When there is more than 1 task, the group_overloaded case already
	 * takes care of cpu with reduced capacity
	 */
	if (rq->cfs.h_nr_runnable != 1)
		return false;

	return check_cpu_capacity(rq, sd);
}

/*
 * update_sg_lb_stats - 汇总组内各 CPU 的负载、任务数和空闲容量，
 * 并确定该组均衡类型。
 */
/**
 * update_sg_lb_stats - Update sched_group's statistics for load balancing.
 * @env: The load balancing environment.
 * @sds: Load-balancing data with statistics of the local group.
 * @group: sched_group whose statistics are to be updated.
 * @sgs: variable to hold the statistics for this group.
 * @sg_overloaded: sched_group is overloaded
 * @sg_overutilized: sched_group is overutilized
 */
static inline void update_sg_lb_stats(struct lb_env *env,
				      struct sd_lb_stats *sds,
				      struct sched_group *group,
				      struct sg_lb_stats *sgs,
				      bool *sg_overloaded,
				      bool *sg_overutilized)
{
	int i, nr_running, local_group, sd_flags = env->sd->flags;
	bool balancing_at_rd = !env->sd->parent;

	memset(sgs, 0, sizeof(*sgs));

	local_group = group == sds->local;

	for_each_cpu_and(i, sched_group_span(group), env->cpus) {
		struct rq *rq = cpu_rq(i);
		unsigned long load = cpu_load(rq);

		sgs->group_load += load;
		sgs->group_util += cpu_util_cfs(i);
		sgs->group_runnable += cpu_runnable(rq);
		sgs->sum_h_nr_running += rq->cfs.h_nr_runnable;

		nr_running = rq->nr_running;
		sgs->sum_nr_running += nr_running;

		if (cpu_overutilized(i))
			*sg_overutilized = 1;

		/*
		 * No need to call idle_cpu() if nr_running is not 0
		 */
		if (!nr_running && idle_cpu(i)) {
			sgs->idle_cpus++;
			/* Idle cpu can't have misfit task */
			continue;
		}

		/* Overload indicator is only updated at root domain */
		if (balancing_at_rd && nr_running > 1)
			*sg_overloaded = 1;

#ifdef CONFIG_NUMA_BALANCING
		/* Only fbq_classify_group() uses this to classify NUMA groups */
		if (sd_flags & SD_NUMA) {
			sgs->nr_numa_running += rq->nr_numa_running;
			sgs->nr_preferred_running += rq->nr_preferred_running;
		}
#endif
		if (local_group)
			continue;

		if (sd_flags & SD_ASYM_CPUCAPACITY) {
			/* Check for a misfit task on the cpu */
			if (sgs->group_misfit_task_load < rq->misfit_task_load) {
				sgs->group_misfit_task_load = rq->misfit_task_load;

				if (balancing_at_rd)
					*sg_overloaded = 1;
			}
		} else if (env->idle && sched_reduced_capacity(rq, env->sd)) {
			/* Check for a task running on a CPU with reduced capacity */
			if (sgs->group_misfit_task_load < load)
				sgs->group_misfit_task_load = load;
		}
	}

	sgs->group_capacity = group->sgc->capacity;

	sgs->group_weight = group->group_weight;

	/* Check if dst CPU is idle and preferred to this group */
	if (!local_group && env->idle && sgs->sum_h_nr_running &&
	    sched_group_asym(env, sgs, group))
		sgs->group_asym_packing = 1;

	/* Check for loaded SMT group to be balanced to dst CPU */
	if (!local_group && smt_balance(env, sgs, group))
		sgs->group_smt_balance = 1;

	sgs->group_type = group_classify(env->sd->imbalance_pct, group, sgs);

	/* Computing avg_load makes sense only when group is overloaded */
	if (sgs->group_type == group_overloaded)
		sgs->avg_load = (sgs->group_load * SCHED_CAPACITY_SCALE) /
				sgs->group_capacity;
}

/*
 * update_sd_pick_busiest - 比较候选组是否比已选组更适合提供迁移任务。
 * 先比较组类型，再结合容量、SMT、非对称优先级和负载细分。
 */
/**
 * update_sd_pick_busiest - return 1 on busiest group
 * @env: The load balancing environment.
 * @sds: sched_domain statistics
 * @sg: sched_group candidate to be checked for being the busiest
 * @sgs: sched_group statistics
 *
 * Determine if @sg is a busier group than the previously selected
 * busiest group.
 *
 * Return: %true if @sg is a busier group than the previously selected
 * busiest group. %false otherwise.
 */
static bool update_sd_pick_busiest(struct lb_env *env,
				   struct sd_lb_stats *sds,
				   struct sched_group *sg,
				   struct sg_lb_stats *sgs)
{
	struct sg_lb_stats *busiest = &sds->busiest_stat;

	/* Make sure that there is at least one task to pull */
	if (!sgs->sum_h_nr_running)
		return false;

	/*
	 * Don't try to pull misfit tasks we can't help.
	 * We can use max_capacity here as reduction in capacity on some
	 * CPUs in the group should either be possible to resolve
	 * internally or be covered by avg_load imbalance (eventually).
	 *
	 * When SMT is active, only pull a misfit to dst_cpu if it is on a
	 * fully idle core; otherwise the effective capacity of the core is
	 * reduced and we may not actually provide more capacity than the
	 * source.
	 */
	if ((env->sd->flags & SD_ASYM_CPUCAPACITY) &&
	    (sgs->group_type == group_misfit_task) &&
	    (!env->dst_core_idle ||
	     !capacity_greater(capacity_of(env->dst_cpu), sg->sgc->max_capacity) ||
	     sds->local_stat.group_type != group_has_spare))
		return false;

	/*
	 * Candidate sg has no more than one task per CPU and has higher
	 * per-CPU capacity. Migrating tasks to less capable CPUs may harm
	 * throughput. Maximize throughput, power/energy consequences are not
	 * considered.
	 */
	if ((env->sd->flags & SD_ASYM_CPUCAPACITY) &&
	    (sgs->group_type <= group_fully_busy) &&
	    (capacity_greater(sg->sgc->min_capacity, capacity_of(env->dst_cpu))))
		return false;

	if (sgs->group_type > busiest->group_type)
		return true;

	if (sgs->group_type < busiest->group_type)
		return false;

	/*
	 * The candidate and the current busiest group are the same type of
	 * group. Let check which one is the busiest according to the type.
	 */

	switch (sgs->group_type) {
	case group_overloaded:
		/* Select the overloaded group with highest avg_load. */
		return sgs->avg_load > busiest->avg_load;

	case group_imbalanced:
		/*
		 * Select the 1st imbalanced group as we don't have any way to
		 * choose one more than another.
		 */
		return false;

	case group_asym_packing:
		/* Prefer to move from lowest priority CPU's work */
		return sched_asym_prefer(READ_ONCE(sds->busiest->asym_prefer_cpu),
					 READ_ONCE(sg->asym_prefer_cpu));

	case group_misfit_task:
		/*
		 * If we have more than one misfit sg go with the biggest
		 * misfit.
		 */
		return sgs->group_misfit_task_load > busiest->group_misfit_task_load;

	case group_smt_balance:
		/*
		 * Check if we have spare CPUs on either SMT group to
		 * choose has spare or fully busy handling.
		 */
		if (sgs->idle_cpus != 0 || busiest->idle_cpus != 0)
			goto has_spare;

		fallthrough;

	case group_fully_busy:
		/*
		 * Select the fully busy group with highest avg_load. In
		 * theory, there is no need to pull task from such kind of
		 * group because tasks have all compute capacity that they need
		 * but we can still improve the overall throughput by reducing
		 * contention when accessing shared HW resources.
		 *
		 * XXX for now avg_load is not computed and always 0 so we
		 * select the 1st one, except if @sg is composed of SMT
		 * siblings.
		 */

		if (sgs->avg_load < busiest->avg_load)
			return false;

		if (sgs->avg_load == busiest->avg_load) {
			/*
			 * SMT sched groups need more help than non-SMT groups.
			 * If @sg happens to also be SMT, either choice is good.
			 */
			if (sds->busiest->flags & SD_SHARE_CPUCAPACITY)
				return false;
		}

		break;

	case group_has_spare:
		/*
		 * Do not pick sg with SMT CPUs over sg with pure CPUs,
		 * as we do not want to pull task off SMT core with one task
		 * and make the core idle.
		 */
		if (smt_vs_nonsmt_groups(sds->busiest, sg)) {
			if (sg->flags & SD_SHARE_CPUCAPACITY && sgs->sum_h_nr_running <= 1)
				return false;
			else
				return true;
		}
has_spare:

		/*
		 * Select not overloaded group with lowest number of idle CPUs
		 * and highest number of running tasks. We could also compare
		 * the spare capacity which is more stable but it can end up
		 * that the group has less spare capacity but finally more idle
		 * CPUs which means less opportunity to pull tasks.
		 */
		if (sgs->idle_cpus > busiest->idle_cpus)
			return false;
		else if ((sgs->idle_cpus == busiest->idle_cpus) &&
			 (sgs->sum_nr_running <= busiest->sum_nr_running))
			return false;

		break;
	}

	return true;
}

#ifdef CONFIG_NUMA_BALANCING
/*
 * fbq_classify_group - 按 NUMA 归属将组分级，
 * 优先寻找无固定偏好或偏离首选节点的任务。
 */
static inline enum fbq_type fbq_classify_group(struct sg_lb_stats *sgs)
{
	if (sgs->sum_h_nr_running > sgs->nr_numa_running)
		return regular;
	if (sgs->sum_h_nr_running > sgs->nr_preferred_running)
		return remote;
	return all;
}

/*
 * fbq_classify_rq - 按运行队列任务的 NUMA 分布返回对应的候选来源级别。
 */
static inline enum fbq_type fbq_classify_rq(struct rq *rq)
{
	if (rq->nr_running > rq->nr_numa_running)
		return regular;
	if (rq->nr_running > rq->nr_preferred_running)
		return remote;
	return all;
}
#else /* !CONFIG_NUMA_BALANCING: */
/*
 * fbq_classify_group - 未启用 NUMA 自动均衡时，允许考虑所有类型的源队列。
 */
static inline enum fbq_type fbq_classify_group(struct sg_lb_stats *sgs)
{
	return all;
}

/*
 * fbq_classify_rq - 未启用 NUMA 自动均衡时，把运行队列视为普通来源。
 */
static inline enum fbq_type fbq_classify_rq(struct rq *rq)
{
	return regular;
}
#endif /* !CONFIG_NUMA_BALANCING */


struct sg_lb_stats;

/*
 * task_running_on_cpu - return 1 if @p is running on @cpu.
 */

/*
 * task_running_on_cpu - 判断任务是否已计入指定 CPU 的排队数，
 * 供选核统计扣除自身。
 * 这里检查 queued 状态，不表示任务此刻一定正在执行指令。
 */
static unsigned int task_running_on_cpu(int cpu, struct task_struct *p)
{
	/* Task has no contribution or is new */
	if (cpu != task_cpu(p) || !READ_ONCE(p->se.avg.last_update_time))
		return 0;

	if (task_on_rq_queued(p))
		return 1;

	return 0;
}

/*
 * idle_cpu_without - 在已排除任务 p 的队列计数基础上，
 * 检查执行者及待处理唤醒是否允许视为空闲。
 */
/**
 * idle_cpu_without - would a given CPU be idle without p ?
 * @cpu: the processor on which idleness is tested.
 * @p: task which should be ignored.
 *
 * Return: 1 if the CPU would be idle. 0 otherwise.
 */
static int idle_cpu_without(int cpu, struct task_struct *p)
{
	struct rq *rq = cpu_rq(cpu);

	if (rq->curr != rq->idle && rq->curr != p)
		return 0;

	/*
	 * rq->nr_running can't be used but an updated version without the
	 * impact of p on cpu must be used instead. The updated nr_running
	 * be computed and tested before calling idle_cpu_without().
	 */

	if (rq->ttwu_pending)
		return 0;

	return 1;
}

/*
 * update_sg_wakeup_stats - Update sched_group's statistics for wakeup.
 * @sd: The sched_domain level to look for idlest group.
 * @group: sched_group whose statistics are to be updated.
 * @sgs: variable to hold the statistics for this group.
 * @p: The task for which we look for the idlest group/CPU.
 */
/*
 * update_sg_wakeup_stats - 统计候选组在移除待放置任务后的状态，
 * 避免把任务自身当作额外负载。
 */
static inline void update_sg_wakeup_stats(struct sched_domain *sd,
					  struct sched_group *group,
					  struct sg_lb_stats *sgs,
					  struct task_struct *p)
{
	int i, nr_running;

	memset(sgs, 0, sizeof(*sgs));

	/* Assume that task can't fit any CPU of the group */
	if (sd->flags & SD_ASYM_CPUCAPACITY)
		sgs->group_misfit_task_load = 1;

	for_each_cpu_and(i, sched_group_span(group), p->cpus_ptr) {
		struct rq *rq = cpu_rq(i);
		unsigned int local;

		sgs->group_load += cpu_load_without(rq, p);
		sgs->group_util += cpu_util_without(i, p);
		sgs->group_runnable += cpu_runnable_without(rq, p);
		local = task_running_on_cpu(i, p);
		sgs->sum_h_nr_running += rq->cfs.h_nr_runnable - local;

		nr_running = rq->nr_running - local;
		sgs->sum_nr_running += nr_running;

		/*
		 * No need to call idle_cpu_without() if nr_running is not 0
		 */
		if (!nr_running && idle_cpu_without(i, p))
			sgs->idle_cpus++;

		/* Check if task fits in the CPU */
		if (sd->flags & SD_ASYM_CPUCAPACITY &&
		    sgs->group_misfit_task_load &&
		    task_fits_cpu(p, i))
			sgs->group_misfit_task_load = 0;

	}

	sgs->group_capacity = group->sgc->capacity;

	sgs->group_weight = group->group_weight;

	sgs->group_type = group_classify(sd->imbalance_pct, group, sgs);

	/*
	 * Computing avg_load makes sense only when group is fully busy or
	 * overloaded
	 */
	if (sgs->group_type == group_fully_busy ||
		sgs->group_type == group_overloaded)
		sgs->avg_load = (sgs->group_load * SCHED_CAPACITY_SCALE) /
				sgs->group_capacity;
}

/*
 * update_pick_idlest：
 * 按组类型及容量、空闲 CPU 数、利用率比较更合适的目标组。
 */
static bool update_pick_idlest(struct sched_group *idlest,
			       struct sg_lb_stats *idlest_sgs,
			       struct sched_group *group,
			       struct sg_lb_stats *sgs)
{
	if (sgs->group_type < idlest_sgs->group_type)
		return true;

	if (sgs->group_type > idlest_sgs->group_type)
		return false;

	/*
	 * The candidate and the current idlest group are the same type of
	 * group. Let check which one is the idlest according to the type.
	 */

	switch (sgs->group_type) {
	case group_overloaded:
	case group_fully_busy:
		/* Select the group with lowest avg_load. */
		if (idlest_sgs->avg_load <= sgs->avg_load)
			return false;
		break;

	case group_imbalanced:
	case group_asym_packing:
	case group_smt_balance:
		/* Those types are not used in the slow wakeup path */
		return false;

	case group_misfit_task:
		/* Select group with the highest max capacity */
		if (idlest->sgc->max_capacity >= group->sgc->max_capacity)
			return false;
		break;

	case group_has_spare:
		/* Select group with most idle CPUs */
		if (idlest_sgs->idle_cpus > sgs->idle_cpus)
			return false;

		/* Select group with lowest group_util */
		if (idlest_sgs->idle_cpus == sgs->idle_cpus &&
			idlest_sgs->group_util <= sgs->group_util)
			return false;

		break;
	}

	return true;
}

/*
 * sched_balance_find_dst_group() finds and returns the least busy CPU group within the
 * domain.
 *
 * Assumes p is allowed on at least one CPU in sd.
 */
/*
 * sched_balance_find_dst_group - 寻找更合适的目标 CPU 组，
 * 并判断迁离本组是否值得。
 * 同时考虑亲和性、核心 cookie、NUMA 偏好和不均衡容差。
 */
static struct sched_group *
sched_balance_find_dst_group(struct sched_domain *sd, struct task_struct *p, int this_cpu)
{
	struct sched_group *idlest = NULL, *local = NULL, *group = sd->groups;
	struct sg_lb_stats local_sgs, tmp_sgs;
	struct sg_lb_stats *sgs;
	unsigned long imbalance;
	struct sg_lb_stats idlest_sgs = {
			.avg_load = UINT_MAX,
			.group_type = group_overloaded,
	};

	do {
		int local_group;

		/* Skip over this group if it has no CPUs allowed */
		if (!cpumask_intersects(sched_group_span(group),
					p->cpus_ptr))
			continue;

		/* Skip over this group if no cookie matched */
		if (!sched_group_cookie_match(cpu_rq(this_cpu), p, group))
			continue;

		local_group = cpumask_test_cpu(this_cpu,
					       sched_group_span(group));

		if (local_group) {
			sgs = &local_sgs;
			local = group;
		} else {
			sgs = &tmp_sgs;
		}

		update_sg_wakeup_stats(sd, group, sgs, p);

		if (!local_group && update_pick_idlest(idlest, &idlest_sgs, group, sgs)) {
			idlest = group;
			idlest_sgs = *sgs;
		}

	} while (group = group->next, group != sd->groups);


	/* There is no idlest group to push tasks to */
	if (!idlest)
		return NULL;

	/* The local group has been skipped because of CPU affinity */
	if (!local)
		return idlest;

	/*
	 * If the local group is idler than the selected idlest group
	 * don't try and push the task.
	 */
	if (local_sgs.group_type < idlest_sgs.group_type)
		return NULL;

	/*
	 * If the local group is busier than the selected idlest group
	 * try and push the task.
	 */
	if (local_sgs.group_type > idlest_sgs.group_type)
		return idlest;

	switch (local_sgs.group_type) {
	case group_overloaded:
	case group_fully_busy:

		/* Calculate allowed imbalance based on load */
		imbalance = scale_load_down(NICE_0_LOAD) *
				(sd->imbalance_pct-100) / 100;

		/*
		 * When comparing groups across NUMA domains, it's possible for
		 * the local domain to be very lightly loaded relative to the
		 * remote domains but "imbalance" skews the comparison making
		 * remote CPUs look much more favourable. When considering
		 * cross-domain, add imbalance to the load on the remote node
		 * and consider staying local.
		 */

		if ((sd->flags & SD_NUMA) &&
		    ((idlest_sgs.avg_load + imbalance) >= local_sgs.avg_load))
			return NULL;

		/*
		 * If the local group is less loaded than the selected
		 * idlest group don't try and push any tasks.
		 */
		if (idlest_sgs.avg_load >= (local_sgs.avg_load + imbalance))
			return NULL;

		if (100 * local_sgs.avg_load <= sd->imbalance_pct * idlest_sgs.avg_load)
			return NULL;
		break;

	case group_imbalanced:
	case group_asym_packing:
	case group_smt_balance:
		/* Those type are not used in the slow wakeup path */
		return NULL;

	case group_misfit_task:
		/* Select group with the highest max capacity */
		if (local->sgc->max_capacity >= idlest->sgc->max_capacity)
			return NULL;
		break;

	case group_has_spare:
#ifdef CONFIG_NUMA
		if (sd->flags & SD_NUMA) {
			int imb_numa_nr = sd->imb_numa_nr;
#ifdef CONFIG_NUMA_BALANCING
			int idlest_cpu;
			/*
			 * If there is spare capacity at NUMA, try to select
			 * the preferred node
			 */
			if (cpu_to_node(this_cpu) == p->numa_preferred_nid)
				return NULL;

			idlest_cpu = cpumask_first(sched_group_span(idlest));
			if (cpu_to_node(idlest_cpu) == p->numa_preferred_nid)
				return idlest;
#endif /* CONFIG_NUMA_BALANCING */
			/*
			 * Otherwise, keep the task close to the wakeup source
			 * and improve locality if the number of running tasks
			 * would remain below threshold where an imbalance is
			 * allowed while accounting for the possibility the
			 * task is pinned to a subset of CPUs. If there is a
			 * real need of migration, periodic load balance will
			 * take care of it.
			 */
			if (p->nr_cpus_allowed != NR_CPUS) {
				struct cpumask *cpus = this_cpu_cpumask_var_ptr(select_rq_mask);

				cpumask_and(cpus, sched_group_span(local), p->cpus_ptr);
				imb_numa_nr = min(cpumask_weight(cpus), sd->imb_numa_nr);
			}

			imbalance = abs(local_sgs.idle_cpus - idlest_sgs.idle_cpus);
			if (!adjust_numa_imbalance(imbalance,
						   local_sgs.sum_nr_running + 1,
						   imb_numa_nr)) {
				return NULL;
			}
		}
#endif /* CONFIG_NUMA */

		/*
		 * Select group with highest number of idle CPUs. We could also
		 * compare the utilization which is more stable but it can end
		 * up that the group has less spare capacity but finally more
		 * idle CPUs which means more opportunity to run task.
		 */
		if (local_sgs.idle_cpus >= idlest_sgs.idle_cpus)
			return NULL;
		break;
	}

	return idlest;
}

/*
 * update_idle_cpu_scan - 根据共享缓存域利用率动态缩减空闲 CPU 搜索预算，
 * 降低高负载下的扫描成本。
 */
static void update_idle_cpu_scan(struct lb_env *env,
				 unsigned long sum_util)
{
	struct sched_domain_shared *sd_share;
	int llc_weight, pct;
	u64 x, y, tmp;
	/*
	 * Update the number of CPUs to scan in LLC domain, which could
	 * be used as a hint in select_idle_cpu(). The update of sd_share
	 * could be expensive because it is within a shared cache line.
	 * So the write of this hint only occurs during periodic load
	 * balancing, rather than CPU_NEWLY_IDLE, because the latter
	 * can fire way more frequently than the former.
	 */
	if (!sched_feat(SIS_UTIL) || env->idle == CPU_NEWLY_IDLE)
		return;

	llc_weight = per_cpu(sd_llc_size, env->dst_cpu);
	if (env->sd->span_weight != llc_weight)
		return;

	sd_share = rcu_dereference(per_cpu(sd_llc_shared, env->dst_cpu));
	if (!sd_share)
		return;

	/*
	 * The number of CPUs to search drops as sum_util increases, when
	 * sum_util hits 85% or above, the scan stops.
	 * The reason to choose 85% as the threshold is because this is the
	 * imbalance_pct(117) when a LLC sched group is overloaded.
	 *
	 * let y = SCHED_CAPACITY_SCALE - p * x^2                       [1]
	 * and y'= y / SCHED_CAPACITY_SCALE
	 *
	 * x is the ratio of sum_util compared to the CPU capacity:
	 * x = sum_util / (llc_weight * SCHED_CAPACITY_SCALE)
	 * y' is the ratio of CPUs to be scanned in the LLC domain,
	 * and the number of CPUs to scan is calculated by:
	 *
	 * nr_scan = llc_weight * y'                                    [2]
	 *
	 * When x hits the threshold of overloaded, AKA, when
	 * x = 100 / pct, y drops to 0. According to [1],
	 * p should be SCHED_CAPACITY_SCALE * pct^2 / 10000
	 *
	 * Scale x by SCHED_CAPACITY_SCALE:
	 * x' = sum_util / llc_weight;                                  [3]
	 *
	 * and finally [1] becomes:
	 * y = SCHED_CAPACITY_SCALE -
	 *     x'^2 * pct^2 / (10000 * SCHED_CAPACITY_SCALE)            [4]
	 *
	 */
	/* equation [3] */
	x = sum_util;
	do_div(x, llc_weight);

	/* equation [4] */
	pct = env->sd->imbalance_pct;
	tmp = x * x * pct * pct;
	do_div(tmp, 10000 * SCHED_CAPACITY_SCALE);
	tmp = min_t(long, tmp, SCHED_CAPACITY_SCALE);
	y = SCHED_CAPACITY_SCALE - tmp;

	/* equation [2] */
	y *= llc_weight;
	do_div(y, SCHED_CAPACITY_SCALE);
	if ((int)y != sd_share->nr_idle_scan)
		WRITE_ONCE(sd_share->nr_idle_scan, (int)y);
}

/*
 * update_sd_lb_stats - 汇总调度域各组状态，选择最忙组，
 * 并更新根域负载标记和空闲扫描预算。
 */
/**
 * update_sd_lb_stats - Update sched_domain's statistics for load balancing.
 * @env: The load balancing environment.
 * @sds: variable to hold the statistics for this sched_domain.
 */

static inline void update_sd_lb_stats(struct lb_env *env, struct sd_lb_stats *sds)
{
	struct sched_group *sg = env->sd->groups;
	struct sg_lb_stats *local = &sds->local_stat;
	struct sg_lb_stats tmp_sgs;
	unsigned long sum_util = 0;
	bool sg_overloaded = 0, sg_overutilized = 0;

	env->dst_core_idle = !sched_smt_active() || is_core_idle(env->dst_cpu);

	do {
		struct sg_lb_stats *sgs = &tmp_sgs;
		int local_group;

		local_group = cpumask_test_cpu(env->dst_cpu, sched_group_span(sg));
		if (local_group) {
			sds->local = sg;
			sgs = local;

			if (env->idle != CPU_NEWLY_IDLE ||
			    time_after_eq(jiffies, sg->sgc->next_update))
				update_group_capacity(env->sd, env->dst_cpu);
		}

		update_sg_lb_stats(env, sds, sg, sgs, &sg_overloaded, &sg_overutilized);

		if (!local_group && update_sd_pick_busiest(env, sds, sg, sgs)) {
			sds->busiest = sg;
			sds->busiest_stat = *sgs;
		}

		/* Now, start updating sd_lb_stats */
		sds->total_load += sgs->group_load;
		sds->total_capacity += sgs->group_capacity;

		sum_util += sgs->group_util;
		sg = sg->next;
	} while (sg != env->sd->groups);

	/*
	 * Indicate that the child domain of the busiest group prefers tasks
	 * go to a child's sibling domains first. NB the flags of a sched group
	 * are those of the child domain.
	 */
	if (sds->busiest)
		sds->prefer_sibling = !!(sds->busiest->flags & SD_PREFER_SIBLING);


	if (env->sd->flags & SD_NUMA)
		env->fbq_type = fbq_classify_group(&sds->busiest_stat);

	if (!env->sd->parent) {
		/* update overload indicator if we are at root domain */
		set_rd_overloaded(env->dst_rq->rd, sg_overloaded);

		/* Update over-utilization (tipping point, U >= 0) indicator */
		set_rd_overutilized(env->dst_rq->rd, sg_overutilized);
	} else if (sg_overutilized) {
		set_rd_overutilized(env->dst_rq->rd, sg_overutilized);
	}

	update_idle_cpu_scan(env, sum_util);
}

/*
 * calculate_imbalance - 根据源组与本地组状态，
 * 确定迁移类型及需要搬移的数量。
 * imbalance 的单位由 migration_type 决定，可能是负载、利用率或任务数。
 */
/**
 * calculate_imbalance - Calculate the amount of imbalance present within the
 *			 groups of a given sched_domain during load balance.
 * @env: load balance environment
 * @sds: statistics of the sched_domain whose imbalance is to be calculated.
 */
static inline void calculate_imbalance(struct lb_env *env, struct sd_lb_stats *sds)
{
	struct sg_lb_stats *local, *busiest;

	local = &sds->local_stat;
	busiest = &sds->busiest_stat;

	if (busiest->group_type == group_misfit_task) {
		if (env->sd->flags & SD_ASYM_CPUCAPACITY) {
			/* Set imbalance to allow misfit tasks to be balanced. */
			env->migration_type = migrate_misfit;
			env->imbalance = 1;
		} else {
			/*
			 * Set load imbalance to allow moving task from cpu
			 * with reduced capacity.
			 */
			env->migration_type = migrate_load;
			env->imbalance = busiest->group_misfit_task_load;
		}
		return;
	}

	if (busiest->group_type == group_asym_packing) {
		/*
		 * In case of asym capacity, we will try to migrate all load to
		 * the preferred CPU.
		 */
		env->migration_type = migrate_task;
		env->imbalance = busiest->sum_h_nr_running;
		return;
	}

	if (busiest->group_type == group_smt_balance) {
		/* Reduce number of tasks sharing CPU capacity */
		env->migration_type = migrate_task;
		env->imbalance = 1;
		return;
	}

	if (busiest->group_type == group_imbalanced) {
		/*
		 * In the group_imb case we cannot rely on group-wide averages
		 * to ensure CPU-load equilibrium, try to move any task to fix
		 * the imbalance. The next load balance will take care of
		 * balancing back the system.
		 */
		env->migration_type = migrate_task;
		env->imbalance = 1;
		return;
	}

	/*
	 * Try to use spare capacity of local group without overloading it or
	 * emptying busiest.
	 */
	if (local->group_type == group_has_spare) {
		if ((busiest->group_type > group_fully_busy) &&
		    !(env->sd->flags & SD_SHARE_LLC)) {
			/*
			 * If busiest is overloaded, try to fill spare
			 * capacity. This might end up creating spare capacity
			 * in busiest or busiest still being overloaded but
			 * there is no simple way to directly compute the
			 * amount of load to migrate in order to balance the
			 * system.
			 */
			env->migration_type = migrate_util;
			env->imbalance = max(local->group_capacity, local->group_util) -
					 local->group_util;

			/*
			 * In some cases, the group's utilization is max or even
			 * higher than capacity because of migrations but the
			 * local CPU is (newly) idle. There is at least one
			 * waiting task in this overloaded busiest group. Let's
			 * try to pull it.
			 */
			if (env->idle && env->imbalance == 0) {
				env->migration_type = migrate_task;
				env->imbalance = 1;
			}

			return;
		}

		if (busiest->group_weight == 1 || sds->prefer_sibling) {
			/*
			 * When prefer sibling, evenly spread running tasks on
			 * groups.
			 */
			env->migration_type = migrate_task;
			env->imbalance = sibling_imbalance(env, sds, busiest, local);
		} else {

			/*
			 * If there is no overload, we just want to even the number of
			 * idle CPUs.
			 */
			env->migration_type = migrate_task;
			env->imbalance = max_t(long, 0,
					       (local->idle_cpus - busiest->idle_cpus));
		}

#ifdef CONFIG_NUMA
		/* Consider allowing a small imbalance between NUMA groups */
		if (env->sd->flags & SD_NUMA) {
			env->imbalance = adjust_numa_imbalance(env->imbalance,
							       local->sum_nr_running + 1,
							       env->sd->imb_numa_nr);
		}
#endif

		/* Number of tasks to move to restore balance */
		env->imbalance >>= 1;

		return;
	}

	/*
	 * Local is fully busy but has to take more load to relieve the
	 * busiest group
	 */
	if (local->group_type < group_overloaded) {
		/*
		 * Local will become overloaded so the avg_load metrics are
		 * finally needed.
		 */

		local->avg_load = (local->group_load * SCHED_CAPACITY_SCALE) /
				  local->group_capacity;

		/*
		 * If the local group is more loaded than the selected
		 * busiest group don't try to pull any tasks.
		 */
		if (local->avg_load >= busiest->avg_load) {
			env->imbalance = 0;
			return;
		}

		sds->avg_load = (sds->total_load * SCHED_CAPACITY_SCALE) /
				sds->total_capacity;

		/*
		 * If the local group is more loaded than the average system
		 * load, don't try to pull any tasks.
		 */
		if (local->avg_load >= sds->avg_load) {
			env->imbalance = 0;
			return;
		}

	}

	/*
	 * Both group are or will become overloaded and we're trying to get all
	 * the CPUs to the average_load, so we don't want to push ourselves
	 * above the average load, nor do we wish to reduce the max loaded CPU
	 * below the average load. At the same time, we also don't want to
	 * reduce the group load below the group capacity. Thus we look for
	 * the minimum possible imbalance.
	 */
	env->migration_type = migrate_load;
	env->imbalance = min(
		(busiest->avg_load - sds->avg_load) * busiest->group_capacity,
		(sds->avg_load - local->avg_load) * local->group_capacity
	) / SCHED_CAPACITY_SCALE;
}

/******* sched_balance_find_src_group() helpers end here *********************/

/*
 * Decision matrix according to the local and busiest group type:
 *
 * busiest \ local has_spare fully_busy misfit asym imbalanced overloaded
 * has_spare        nr_idle   balanced   N/A    N/A  balanced   balanced
 * fully_busy       nr_idle   nr_idle    N/A    N/A  balanced   balanced
 * misfit_task      force     N/A        N/A    N/A  N/A        N/A
 * asym_packing     force     force      N/A    N/A  force      force
 * imbalanced       force     force      N/A    N/A  force      force
 * overloaded       force     force      N/A    N/A  force      avg_load
 *
 * N/A :      Not Applicable because already filtered while updating
 *            statistics.
 * balanced : The system is balanced for these 2 groups.
 * force :    Calculate the imbalance as load migration is probably needed.
 * avg_load : Only if imbalance is significant enough.
 * nr_idle :  dst_cpu is not busy and the number of idle CPUs is quite
 *            different in groups.
 */

/*
 * sched_balance_find_src_group - 选择值得拉取任务的最忙组，
 * 并计算本轮迁移预算；已均衡则返回 NULL。
 */
/**
 * sched_balance_find_src_group - Returns the busiest group within the sched_domain
 * if there is an imbalance.
 * @env: The load balancing environment.
 *
 * Also calculates the amount of runnable load which should be moved
 * to restore balance.
 *
 * Return:	- The busiest group if imbalance exists.
 */
static struct sched_group *sched_balance_find_src_group(struct lb_env *env)
{
	struct sg_lb_stats *local, *busiest;
	struct sd_lb_stats sds;

	init_sd_lb_stats(&sds);

	/*
	 * Compute the various statistics relevant for load balancing at
	 * this level.
	 */
	update_sd_lb_stats(env, &sds);

	/* There is no busy sibling group to pull tasks from */
	if (!sds.busiest)
		goto out_balanced;

	busiest = &sds.busiest_stat;

	/* Misfit tasks should be dealt with regardless of the avg load */
	if (busiest->group_type == group_misfit_task)
		goto force_balance;

	/*
	 * 能耗模型适用且系统尚未过度利用时，保留能耗选核形成的分布，
	 * 减少无益搬迁。
	 */
	if (!is_rd_overutilized(env->dst_rq->rd) &&
	    rcu_dereference(env->dst_rq->rd->pd))
		goto out_balanced;

	/* ASYM feature bypasses nice load balance check */
	if (busiest->group_type == group_asym_packing)
		goto force_balance;

	/*
	 * If the busiest group is imbalanced the below checks don't
	 * work because they assume all things are equal, which typically
	 * isn't true due to cpus_ptr constraints and the like.
	 */
	if (busiest->group_type == group_imbalanced)
		goto force_balance;

	local = &sds.local_stat;
	/*
	 * If the local group is busier than the selected busiest group
	 * don't try and pull any tasks.
	 */
	if (local->group_type > busiest->group_type)
		goto out_balanced;

	/*
	 * When groups are overloaded, use the avg_load to ensure fairness
	 * between tasks.
	 */
	if (local->group_type == group_overloaded) {
		/*
		 * If the local group is more loaded than the selected
		 * busiest group don't try to pull any tasks.
		 */
		if (local->avg_load >= busiest->avg_load)
			goto out_balanced;

		/* XXX broken for overlapping NUMA groups */
		sds.avg_load = (sds.total_load * SCHED_CAPACITY_SCALE) /
				sds.total_capacity;

		/*
		 * Don't pull any tasks if this group is already above the
		 * domain average load.
		 */
		if (local->avg_load >= sds.avg_load)
			goto out_balanced;

		/*
		 * If the busiest group is more loaded, use imbalance_pct to be
		 * conservative.
		 */
		if (100 * busiest->avg_load <=
				env->sd->imbalance_pct * local->avg_load)
			goto out_balanced;
	}

	/*
	 * Try to move all excess tasks to a sibling domain of the busiest
	 * group's child domain.
	 */
	if (sds.prefer_sibling && local->group_type == group_has_spare &&
	    sibling_imbalance(env, &sds, busiest, local) > 1)
		goto force_balance;

	if (busiest->group_type != group_overloaded) {
		if (!env->idle) {
			/*
			 * If the busiest group is not overloaded (and as a
			 * result the local one too) but this CPU is already
			 * busy, let another idle CPU try to pull task.
			 */
			goto out_balanced;
		}

		if (busiest->group_type == group_smt_balance &&
		    smt_vs_nonsmt_groups(sds.local, sds.busiest)) {
			/* Let non SMT CPU pull from SMT CPU sharing with sibling */
			goto force_balance;
		}

		if (busiest->group_weight > 1 &&
		    local->idle_cpus <= (busiest->idle_cpus + 1)) {
			/*
			 * If the busiest group is not overloaded
			 * and there is no imbalance between this and busiest
			 * group wrt idle CPUs, it is balanced. The imbalance
			 * becomes significant if the diff is greater than 1
			 * otherwise we might end up to just move the imbalance
			 * on another group. Of course this applies only if
			 * there is more than 1 CPU per group.
			 */
			goto out_balanced;
		}

		if (busiest->sum_h_nr_running == 1) {
			/*
			 * busiest doesn't have any tasks waiting to run
			 */
			goto out_balanced;
		}
	}

force_balance:
	/* Looks like there is an imbalance. Compute it */
	calculate_imbalance(env, &sds);
	return env->imbalance ? sds.busiest : NULL;

out_balanced:
	env->imbalance = 0;
	return NULL;
}

/*
 * sched_balance_find_src_rq - find the busiest runqueue among the CPUs in the group.
 */
/*
 * sched_balance_find_src_rq - 在最忙组中按迁移类型选择源运行队列，
 * 同时排除不合适的容量和 NUMA 候选。
 */
static struct rq *sched_balance_find_src_rq(struct lb_env *env,
				     struct sched_group *group)
{
	struct rq *busiest = NULL, *rq;
	unsigned long busiest_util = 0, busiest_load = 0, busiest_capacity = 1;
	unsigned int busiest_nr = 0;
	int i;

	for_each_cpu_and(i, sched_group_span(group), env->cpus) {
		unsigned long capacity, load, util;
		unsigned int nr_running;
		enum fbq_type rt;

		rq = cpu_rq(i);
		rt = fbq_classify_rq(rq);

		/*
		 * We classify groups/runqueues into three groups:
		 *  - regular: there are !numa tasks
		 *  - remote:  there are numa tasks that run on the 'wrong' node
		 *  - all:     there is no distinction
		 *
		 * In order to avoid migrating ideally placed numa tasks,
		 * ignore those when there's better options.
		 *
		 * If we ignore the actual busiest queue to migrate another
		 * task, the next balance pass can still reduce the busiest
		 * queue by moving tasks around inside the node.
		 *
		 * If we cannot move enough load due to this classification
		 * the next pass will adjust the group classification and
		 * allow migration of more tasks.
		 *
		 * Both cases only affect the total convergence complexity.
		 */
		if (rt > env->fbq_type)
			continue;

		nr_running = rq->cfs.h_nr_runnable;
		if (!nr_running)
			continue;

		capacity = capacity_of(i);

		/*
		 * For ASYM_CPUCAPACITY domains, don't pick a CPU that could
		 * eventually lead to active_balancing high->low capacity.
		 * Higher per-CPU capacity is considered better than balancing
		 * average load.
		 */
		if (env->sd->flags & SD_ASYM_CPUCAPACITY &&
		    !capacity_greater(capacity_of(env->dst_cpu), capacity) &&
		    nr_running == 1)
			continue;

		/*
		 * Make sure we only pull tasks from a CPU of lower priority
		 * when balancing between SMT siblings.
		 *
		 * If balancing between cores, let lower priority CPUs help
		 * SMT cores with more than one busy sibling.
		 */
		if (sched_asym(env->sd, i, env->dst_cpu) && nr_running == 1)
			continue;

		switch (env->migration_type) {
		case migrate_load:
			/*
			 * When comparing with load imbalance, use cpu_load()
			 * which is not scaled with the CPU capacity.
			 */
			load = cpu_load(rq);

			if (nr_running == 1 && load > env->imbalance &&
			    !check_cpu_capacity(rq, env->sd))
				break;

			/*
			 * For the load comparisons with the other CPUs,
			 * consider the cpu_load() scaled with the CPU
			 * capacity, so that the load can be moved away
			 * from the CPU that is potentially running at a
			 * lower capacity.
			 *
			 * Thus we're looking for max(load_i / capacity_i),
			 * crosswise multiplication to rid ourselves of the
			 * division works out to:
			 * load_i * capacity_j > load_j * capacity_i;
			 * where j is our previous maximum.
			 */
			if (load * busiest_capacity > busiest_load * capacity) {
				busiest_load = load;
				busiest_capacity = capacity;
				busiest = rq;
			}
			break;

		case migrate_util:
			util = cpu_util_cfs_boost(i);

			/*
			 * Don't try to pull utilization from a CPU with one
			 * running task. Whatever its utilization, we will fail
			 * detach the task.
			 */
			if (nr_running <= 1)
				continue;

			if (busiest_util < util) {
				busiest_util = util;
				busiest = rq;
			}
			break;

		case migrate_task:
			if (busiest_nr < nr_running) {
				busiest_nr = nr_running;
				busiest = rq;
			}
			break;

		case migrate_misfit:
			/*
			 * For ASYM_CPUCAPACITY domains with misfit tasks we
			 * simply seek the "biggest" misfit task.
			 */
			if (rq->misfit_task_load > busiest_load) {
				busiest_load = rq->misfit_task_load;
				busiest = rq;
			}

			break;

		}
	}

	return busiest;
}

/*
 * Max backoff if we encounter pinned tasks. Pretty arbitrary value, but
 * so long as it is large enough.
 */
#define MAX_PINNED_INTERVAL	512

/*
 * asym_active_balance：
 * 判断空闲目标 CPU 是否需要通过主动迁移满足非对称打包偏好。
 */
static inline bool
asym_active_balance(struct lb_env *env)
{
	/*
	 * ASYM_PACKING needs to force migrate tasks from busy but lower
	 * priority CPUs in order to pack all tasks in the highest priority
	 * CPUs. When done between cores, do it only if the whole core if the
	 * whole core is idle.
	 *
	 * If @env::src_cpu is an SMT core with busy siblings, let
	 * the lower priority @env::dst_cpu help it. Do not follow
	 * CPU priority.
	 */
	return env->idle && sched_use_asym_prio(env->sd, env->dst_cpu) &&
	       (sched_asym_prefer(env->dst_cpu, env->src_cpu) ||
		!sched_use_asym_prio(env->sd, env->src_cpu));
}

/*
 * imbalanced_active_balance - 按任务数均衡连续失败足够多次时，
 * 允许升级为主动迁移。
 */
static inline bool
imbalanced_active_balance(struct lb_env *env)
{
	struct sched_domain *sd = env->sd;

	/*
	 * The imbalanced case includes the case of pinned tasks preventing a fair
	 * distribution of the load on the system but also the even distribution of the
	 * threads on a system with spare capacity
	 */
	if ((env->migration_type == migrate_task) &&
	    (sd->nr_balance_failed > sd->cache_nice_tries+2))
		return 1;

	return 0;
}

/*
 * need_active_balance - 综合非对称打包、失败次数和容量不匹配，
 * 判断是否需要 CPU stopper 介入。
 */
static int need_active_balance(struct lb_env *env)
{
	struct sched_domain *sd = env->sd;

	if (asym_active_balance(env))
		return 1;

	if (imbalanced_active_balance(env))
		return 1;

	/*
	 * The dst_cpu is idle and the src_cpu CPU has only 1 CFS task.
	 * It's worth migrating the task if the src_cpu's capacity is reduced
	 * because of other sched_class or IRQs if more capacity stays
	 * available on dst_cpu.
	 */
	if (env->idle &&
	    (env->src_rq->cfs.h_nr_runnable == 1)) {
		if ((check_cpu_capacity(env->src_rq, sd)) &&
		    (capacity_of(env->src_cpu)*sd->imbalance_pct < capacity_of(env->dst_cpu)*100))
			return 1;
	}

	if (env->migration_type == migrate_misfit)
		return 1;

	return 0;
}

static int active_load_balance_cpu_stop(void *data);

/*
 * should_we_balance - 选出本组本轮负责均衡的 CPU，
 * 避免多个 CPU 重复拉取同一份负载。
 */
static int should_we_balance(struct lb_env *env)
{
	struct cpumask *swb_cpus = this_cpu_cpumask_var_ptr(should_we_balance_tmpmask);
	struct sched_group *sg = env->sd->groups;
	int cpu, idle_smt = -1;

	/*
	 * Ensure the balancing environment is consistent; can happen
	 * when the softirq triggers 'during' hotplug.
	 */
	if (!cpumask_test_cpu(env->dst_cpu, env->cpus))
		return 0;

	/*
	 * In the newly idle case, we will allow all the CPUs
	 * to do the newly idle load balance.
	 *
	 * However, we bail out if we already have tasks or a wakeup pending,
	 * to optimize wakeup latency.
	 */
	if (env->idle == CPU_NEWLY_IDLE) {
		if (env->dst_rq->nr_running > 0 || env->dst_rq->ttwu_pending)
			return 0;
		return 1;
	}

	cpumask_copy(swb_cpus, group_balance_mask(sg));
	/* Try to find first idle CPU */
	for_each_cpu_and(cpu, swb_cpus, env->cpus) {
		if (!idle_cpu(cpu))
			continue;

		/*
		 * Don't balance to idle SMT in busy core right away when
		 * balancing cores, but remember the first idle SMT CPU for
		 * later consideration.  Find CPU on an idle core first.
		 */
		if (!(env->sd->flags & SD_SHARE_CPUCAPACITY) && !is_core_idle(cpu)) {
			if (idle_smt == -1)
				idle_smt = cpu;
			/*
			 * If the core is not idle, and first SMT sibling which is
			 * idle has been found, then its not needed to check other
			 * SMT siblings for idleness:
			 */
#ifdef CONFIG_SCHED_SMT
			cpumask_andnot(swb_cpus, swb_cpus, cpu_smt_mask(cpu));
#endif
			continue;
		}

		/*
		 * Are we the first idle core in a non-SMT domain or higher,
		 * or the first idle CPU in a SMT domain?
		 */
		return cpu == env->dst_cpu;
	}

	/* Are we the first idle CPU with busy siblings? */
	if (idle_smt != -1)
		return idle_smt == env->dst_cpu;

	/* Are we the first CPU of this group ? */
	return group_balance_cpu(sg) == env->dst_cpu;
}

/*
 * update_lb_imbalance_stat - 按迁移预算的实际单位记录不均衡统计。
 */
static void update_lb_imbalance_stat(struct lb_env *env, struct sched_domain *sd,
				     enum cpu_idle_type idle)
{
	if (!schedstat_enabled())
		return;

	switch (env->migration_type) {
	case migrate_load:
		__schedstat_add(sd->lb_imbalance_load[idle], env->imbalance);
		break;
	case migrate_util:
		__schedstat_add(sd->lb_imbalance_util[idle], env->imbalance);
		break;
	case migrate_task:
		__schedstat_add(sd->lb_imbalance_task[idle], env->imbalance);
		break;
	case migrate_misfit:
		__schedstat_add(sd->lb_imbalance_misfit[idle], env->imbalance);
		break;
	}
}

/*
 * This flag serializes load-balancing passes over large domains
 * (above the NODE topology level) - only one load-balancing instance
 * may run at a time, to reduce overhead on very large systems with
 * lots of CPUs and large NUMA distances.
 *
 * - Note that load-balancing passes triggered while another one
 *   is executing are skipped and not re-tried.
 *
 * - Also note that this does not serialize rebalance_domains()
 *   execution, as non-SD_SERIALIZE domains will still be
 *   load-balanced in parallel.
 */
static atomic_t sched_balance_running = ATOMIC_INIT(0);

/*
 * Check this_cpu to ensure it is balanced within domain. Attempt to move
 * tasks if there is an imbalance.
 */
/*
 * sched_balance_rq - 在一个调度域内完成选源、摘取、接入及必要的主动均衡。
 * 迁移受限时可更换源或目标，并按结果调整失败计数与下一次均衡间隔。
 */
static int sched_balance_rq(int this_cpu, struct rq *this_rq,
			struct sched_domain *sd, enum cpu_idle_type idle,
			int *continue_balancing)
{
	int ld_moved, cur_ld_moved, active_balance = 0;
	struct sched_domain *sd_parent = sd->parent;
	struct sched_group *group;
	struct rq *busiest;
	struct rq_flags rf;
	struct cpumask *cpus = this_cpu_cpumask_var_ptr(load_balance_mask);
	struct lb_env env = {
		.sd		= sd,
		.dst_cpu	= this_cpu,
		.dst_rq		= this_rq,
		.dst_grpmask    = group_balance_mask(sd->groups),
		.idle		= idle,
		.loop_break	= SCHED_NR_MIGRATE_BREAK,
		.cpus		= cpus,
		.fbq_type	= all,
		.tasks		= LIST_HEAD_INIT(env.tasks),
	};
	bool need_unlock = false;

	cpumask_and(cpus, sched_domain_span(sd), cpu_active_mask);

	schedstat_inc(sd->lb_count[idle]);

redo:
	if (!should_we_balance(&env)) {
		*continue_balancing = 0;
		goto out_balanced;
	}

	if (!need_unlock && (sd->flags & SD_SERIALIZE)) {
		int zero = 0;
		if (!atomic_try_cmpxchg_acquire(&sched_balance_running, &zero, 1))
			goto out_balanced;

		need_unlock = true;
	}

	group = sched_balance_find_src_group(&env);
	if (!group) {
		schedstat_inc(sd->lb_nobusyg[idle]);
		goto out_balanced;
	}

	busiest = sched_balance_find_src_rq(&env, group);
	if (!busiest) {
		schedstat_inc(sd->lb_nobusyq[idle]);
		goto out_balanced;
	}

	WARN_ON_ONCE(busiest == env.dst_rq);

	update_lb_imbalance_stat(&env, sd, idle);

	env.src_cpu = busiest->cpu;
	env.src_rq = busiest;

	ld_moved = 0;
	/* Clear this flag as soon as we find a pullable task */
	env.flags |= LBF_ALL_PINNED;
	if (busiest->nr_running > 1) {
		/*
		 * Attempt to move tasks. If sched_balance_find_src_group has found
		 * an imbalance but busiest->nr_running <= 1, the group is
		 * still unbalanced. ld_moved simply stays zero, so it is
		 * correctly treated as an imbalance.
		 */
		env.loop_max  = min(sysctl_sched_nr_migrate, busiest->nr_running);

more_balance:
		rq_lock_irqsave(busiest, &rf);
		update_rq_clock(busiest);

		/*
		 * cur_ld_moved - load moved in current iteration
		 * ld_moved     - cumulative load moved across iterations
		 */
		/*
		 * 摘取阶段持源锁；接入阶段改持目标锁，
		 * 临时链表衔接两个阶段。
		 */
		cur_ld_moved = detach_tasks(&env);

		/*
		 * We've detached some tasks from busiest_rq. Every
		 * task is masked "TASK_ON_RQ_MIGRATING", so we can safely
		 * unlock busiest->lock, and we are able to be sure
		 * that nobody can manipulate the tasks in parallel.
		 * See task_rq_lock() family for the details.
		 */

		rq_unlock(busiest, &rf);

		if (cur_ld_moved) {
			attach_tasks(&env);
			ld_moved += cur_ld_moved;
		}

		local_irq_restore(rf.flags);

		/*
		 * 批次达到持锁扫描上限后重新获取源锁继续，
		 * 保留本轮剩余迁移预算。
		 */
		if (env.flags & LBF_NEED_BREAK) {
			env.flags &= ~LBF_NEED_BREAK;
			goto more_balance;
		}

		/*
		 * Revisit (affine) tasks on src_cpu that couldn't be moved to
		 * us and move them to an alternate dst_cpu in our sched_group
		 * where they can run. The upper limit on how many times we
		 * iterate on same src_cpu is dependent on number of CPUs in our
		 * sched_group.
		 *
		 * This changes load balance semantics a bit on who can move
		 * load to a given_cpu. In addition to the given_cpu itself
		 * (or a ilb_cpu acting on its behalf where given_cpu is
		 * nohz-idle), we now have balance_cpu in a position to move
		 * load to given_cpu. In rare situations, this may cause
		 * conflicts (balance_cpu and given_cpu/ilb_cpu deciding
		 * _independently_ and at _same_ time to move some load to
		 * given_cpu) causing excess load to be moved to given_cpu.
		 * This however should not happen so much in practice and
		 * moreover subsequent load balance cycles should correct the
		 * excess load moved.
		 */
		if ((env.flags & LBF_DST_PINNED) && env.imbalance > 0) {

			/* Prevent to re-select dst_cpu via env's CPUs */
			__cpumask_clear_cpu(env.dst_cpu, env.cpus);

			env.dst_rq	 = cpu_rq(env.new_dst_cpu);
			env.dst_cpu	 = env.new_dst_cpu;
			env.flags	&= ~LBF_DST_PINNED;
			env.loop	 = 0;
			env.loop_break	 = SCHED_NR_MIGRATE_BREAK;

			/*
			 * Go back to "more_balance" rather than "redo" since we
			 * need to continue with same src_cpu.
			 */
			goto more_balance;
		}

		/*
		 * We failed to reach balance because of affinity.
		 */
		if (sd_parent) {
			int *group_imbalance = &sd_parent->groups->sgc->imbalance;

			if ((env.flags & LBF_SOME_PINNED) && env.imbalance > 0)
				*group_imbalance = 1;
		}

		/* All tasks on this runqueue were pinned by CPU affinity */
		if (unlikely(env.flags & LBF_ALL_PINNED)) {
			__cpumask_clear_cpu(cpu_of(busiest), cpus);
			/*
			 * Attempting to continue load balancing at the current
			 * sched_domain level only makes sense if there are
			 * active CPUs remaining as possible busiest CPUs to
			 * pull load from which are not contained within the
			 * destination group that is receiving any migrated
			 * load.
			 */
			if (!cpumask_subset(cpus, env.dst_grpmask)) {
				env.loop = 0;
				env.loop_break = SCHED_NR_MIGRATE_BREAK;
				goto redo;
			}
			goto out_all_pinned;
		}
	}

	if (!ld_moved) {
		schedstat_inc(sd->lb_failed[idle]);
		/*
		 * Increment the failure counter only on periodic balance.
		 * We do not want newidle balance, which can be very
		 * frequent, pollute the failure counter causing
		 * excessive cache_hot migrations and active balances.
		 *
		 * Similarly for migration_misfit which is not related to
		 * load/util migration, don't pollute nr_balance_failed.
		 */
		if (idle != CPU_NEWLY_IDLE &&
		    env.migration_type != migrate_misfit)
			sd->nr_balance_failed++;

		/*
		 * 普通拉取没有进展时，
		 * 特定失衡场景可请求源 CPU stopper 执行主动推送。
		 */
		if (need_active_balance(&env)) {
			unsigned long flags;

			raw_spin_rq_lock_irqsave(busiest, flags);

			/*
			 * Don't kick the active_load_balance_cpu_stop,
			 * if the curr task on busiest CPU can't be
			 * moved to this_cpu:
			 */
			if (!cpumask_test_cpu(this_cpu, busiest->curr->cpus_ptr)) {
				raw_spin_rq_unlock_irqrestore(busiest, flags);
				goto out_one_pinned;
			}

			/* Record that we found at least one task that could run on this_cpu */
			env.flags &= ~LBF_ALL_PINNED;

			/*
			 * ->active_balance synchronizes accesses to
			 * ->active_balance_work.  Once set, it's cleared
			 * only after active load balance is finished.
			 */
			if (!busiest->active_balance) {
				busiest->active_balance = 1;
				busiest->push_cpu = this_cpu;
				active_balance = 1;
			}

			preempt_disable();
			raw_spin_rq_unlock_irqrestore(busiest, flags);
			if (active_balance) {
				stop_one_cpu_nowait(cpu_of(busiest),
					active_load_balance_cpu_stop, busiest,
					&busiest->active_balance_work);
			}
			preempt_enable();
		}
	} else {
		sd->nr_balance_failed = 0;
	}

	if (likely(!active_balance) || need_active_balance(&env)) {
		/* We were unbalanced, so reset the balancing interval */
		sd->balance_interval = sd->min_interval;
	}

	goto out;

out_balanced:
	/*
	 * We reach balance although we may have faced some affinity
	 * constraints. Clear the imbalance flag only if other tasks got
	 * a chance to move and fix the imbalance.
	 */
	if (sd_parent && !(env.flags & LBF_ALL_PINNED)) {
		int *group_imbalance = &sd_parent->groups->sgc->imbalance;

		if (*group_imbalance)
			*group_imbalance = 0;
	}

out_all_pinned:
	/*
	 * We reach balance because all tasks are pinned at this level so
	 * we can't migrate them. Let the imbalance flag set so parent level
	 * can try to migrate them.
	 */
	schedstat_inc(sd->lb_balanced[idle]);

	sd->nr_balance_failed = 0;

out_one_pinned:
	ld_moved = 0;

	/*
	 * sched_balance_newidle() disregards balance intervals, so we could
	 * repeatedly reach this code, which would lead to balance_interval
	 * skyrocketing in a short amount of time. Skip the balance_interval
	 * increase logic to avoid that.
	 *
	 * Similarly misfit migration which is not necessarily an indication of
	 * the system being busy and requires lb to backoff to let it settle
	 * down.
	 */
	if (env.idle == CPU_NEWLY_IDLE ||
	    env.migration_type == migrate_misfit)
		goto out;

	/* tune up the balancing interval */
	if ((env.flags & LBF_ALL_PINNED &&
	     sd->balance_interval < MAX_PINNED_INTERVAL) ||
	    sd->balance_interval < sd->max_interval)
		sd->balance_interval *= 2;
out:
	if (need_unlock)
		atomic_set_release(&sched_balance_running, 0);

	return ld_moved;
}

/*
 * get_sd_balance_interval - 根据 CPU 忙闲状态计算并限制下一次调度域均衡间隔。
 */
static inline unsigned long
get_sd_balance_interval(struct sched_domain *sd, int cpu_busy)
{
	unsigned long interval = sd->balance_interval;

	if (cpu_busy)
		interval *= sd->busy_factor;

	/* scale ms to jiffies */
	interval = msecs_to_jiffies(interval);

	/*
	 * Reduce likelihood of busy balancing at higher domains racing with
	 * balancing at lower domains by preventing their balancing periods
	 * from being multiples of each other.
	 */
	if (cpu_busy)
		interval -= 1;

	interval = clamp(interval, 1UL, max_load_balance_interval);

	return interval;
}

/*
 * update_next_balance：
 * 将候选截止时刻收敛为各调度域中最早需要再次均衡的时刻。
 */
static inline void
update_next_balance(struct sched_domain *sd, unsigned long *next_balance)
{
	unsigned long interval, next;

	/* used by idle balance, so cpu_busy = 0 */
	interval = get_sd_balance_interval(sd, 0);
	next = sd->last_balance + interval;

	if (time_after(*next_balance, next))
		*next_balance = next;
}

/*
 * active_load_balance_cpu_stop is run by the CPU stopper. It pushes
 * running tasks off the busiest CPU onto idle CPUs. It requires at
 * least 1 task to be running on each physical CPU where possible, and
 * avoids physical / logical imbalances.
 */
/*
 * active_load_balance_cpu_stop - 在源 CPU stopper 上下文中摘取一个任务，
 * 再接入目标队列。
 * 用于普通拉取无法迁走任务时的主动均衡。
 */
static int active_load_balance_cpu_stop(void *data)
{
	struct rq *busiest_rq = data;
	int busiest_cpu = cpu_of(busiest_rq);
	int target_cpu = busiest_rq->push_cpu;
	struct rq *target_rq = cpu_rq(target_cpu);
	struct sched_domain *sd;
	struct task_struct *p = NULL;
	struct rq_flags rf;

	rq_lock_irq(busiest_rq, &rf);
	/*
	 * Between queueing the stop-work and running it is a hole in which
	 * CPUs can become inactive. We should not move tasks from or to
	 * inactive CPUs.
	 */
	if (!cpu_active(busiest_cpu) || !cpu_active(target_cpu))
		goto out_unlock;

	/* Make sure the requested CPU hasn't gone down in the meantime: */
	if (unlikely(busiest_cpu != smp_processor_id() ||
		     !busiest_rq->active_balance))
		goto out_unlock;

	/* Is there any task to move? */
	if (busiest_rq->nr_running <= 1)
		goto out_unlock;

	/*
	 * This condition is "impossible", if it occurs
	 * we need to fix it. Originally reported by
	 * Bjorn Helgaas on a 128-CPU setup.
	 */
	WARN_ON_ONCE(busiest_rq == target_rq);

	/* Search for an sd spanning us and the target CPU. */
	rcu_read_lock();
	for_each_domain(target_cpu, sd) {
		if (cpumask_test_cpu(busiest_cpu, sched_domain_span(sd)))
			break;
	}

	if (likely(sd)) {
		struct lb_env env = {
			.sd		= sd,
			.dst_cpu	= target_cpu,
			.dst_rq		= target_rq,
			.src_cpu	= busiest_rq->cpu,
			.src_rq		= busiest_rq,
			.idle		= CPU_IDLE,
			.flags		= LBF_ACTIVE_LB,
		};

		schedstat_inc(sd->alb_count);
		update_rq_clock(busiest_rq);

		p = detach_one_task(&env);
		if (p) {
			schedstat_inc(sd->alb_pushed);
			/* Active balancing done, reset the failure counter. */
			sd->nr_balance_failed = 0;
		} else {
			schedstat_inc(sd->alb_failed);
		}
	}
	rcu_read_unlock();
out_unlock:
	busiest_rq->active_balance = 0;
	rq_unlock(busiest_rq, &rf);

	if (p)
		attach_one_task(target_rq, p);

	local_irq_enable();

	return 0;
}

/*
 * Scale the max sched_balance_rq interval with the number of CPUs in the system.
 * This trades load-balance latency on larger machines for less cross talk.
 */
/*
 * update_max_interval - 随在线 CPU 数调整周期均衡的最大间隔。
 */
void update_max_interval(void)
{
	max_load_balance_interval = HZ*num_online_cpus()/10;
}

/*
 * update_newidle_stats - 累积并定期衰减新空闲均衡的尝试和成功统计，
 * 用于概率选择。
 */
static inline void update_newidle_stats(struct sched_domain *sd, unsigned int success)
{
	sd->newidle_call++;
	sd->newidle_success += success;

	if (sd->newidle_call >= 1024) {
		sd->newidle_ratio = sd->newidle_success;
		sd->newidle_call /= 2;
		sd->newidle_success /= 2;
	}
}

/*
 * update_newidle_cost - 维护新空闲均衡的最大耗时估计，
 * 并随时间缓慢降低旧峰值。
 */
static inline bool
update_newidle_cost(struct sched_domain *sd, u64 cost, unsigned int success)
{
	unsigned long next_decay = sd->last_decay_max_lb_cost + HZ;
	unsigned long now = jiffies;

	if (cost)
		update_newidle_stats(sd, success);

	if (cost > sd->max_newidle_lb_cost) {
		/*
		 * Track max cost of a domain to make sure to not delay the
		 * next wakeup on the CPU.
		 */
		sd->max_newidle_lb_cost = cost;
		sd->last_decay_max_lb_cost = now;

	} else if (time_after(now, next_decay)) {
		/*
		 * Decay the newidle max times by ~1% per second to ensure that
		 * it is not outdated and the current max cost is actually
		 * shorter.
		 */
		sd->max_newidle_lb_cost = (sd->max_newidle_lb_cost * 253) / 256;
		sd->last_decay_max_lb_cost = now;
		return true;
	}

	return false;
}

/*
 * It checks each scheduling domain to see if it is due to be balanced,
 * and initiates a balancing operation if so.
 *
 * Balancing parameters are set up in init_sched_domains.
 */
/*
 * sched_balance_domains - 遍历 CPU 的调度域，
 * 对到期的域执行周期均衡并记录最早下次时间。
 */
static void sched_balance_domains(struct rq *rq, enum cpu_idle_type idle)
{
	int continue_balancing = 1;
	int cpu = rq->cpu;
	int busy = idle != CPU_IDLE && !sched_idle_cpu(cpu);
	unsigned long interval;
	struct sched_domain *sd;
	/* Earliest time when we have to do rebalance again */
	unsigned long next_balance = jiffies + 60*HZ;
	int update_next_balance = 0;
	int need_decay = 0;
	u64 max_cost = 0;

	rcu_read_lock();
	for_each_domain(cpu, sd) {
		/*
		 * Decay the newidle max times here because this is a regular
		 * visit to all the domains.
		 */
		need_decay = update_newidle_cost(sd, 0, 0);
		max_cost += sd->max_newidle_lb_cost;

		/*
		 * Stop the load balance at this level. There is another
		 * CPU in our sched group which is doing load balancing more
		 * actively.
		 */
		if (!continue_balancing) {
			if (need_decay)
				continue;
			break;
		}

		interval = get_sd_balance_interval(sd, busy);
		if (time_after_eq(jiffies, sd->last_balance + interval)) {
			if (sched_balance_rq(cpu, rq, sd, idle, &continue_balancing)) {
				/*
				 * The LBF_DST_PINNED logic could have changed
				 * env->dst_cpu, so we can't know our idle
				 * state even if we migrated tasks. Update it.
				 */
				idle = idle_cpu(cpu);
				busy = !idle && !sched_idle_cpu(cpu);
			}
			sd->last_balance = jiffies;
			interval = get_sd_balance_interval(sd, busy);
		}
		if (time_after(next_balance, sd->last_balance + interval)) {
			next_balance = sd->last_balance + interval;
			update_next_balance = 1;
		}
	}
	if (need_decay) {
		/*
		 * Ensure the rq-wide value also decays but keep it at a
		 * reasonable floor to avoid funnies with rq->avg_idle.
		 */
		rq->max_idle_balance_cost =
			max((u64)sysctl_sched_migration_cost, max_cost);
	}
	rcu_read_unlock();

	/*
	 * next_balance will be updated only when there is a need.
	 * When the cpu is attached to null domain for ex, it will not be
	 * updated.
	 */
	if (likely(update_next_balance))
		rq->next_balance = next_balance;

}

/*
 * on_null_domain - 判断 CPU 是否没有可参与负载均衡的调度域。
 */
static inline int on_null_domain(struct rq *rq)
{
	return unlikely(!rcu_dereference_sched(rq->sd));
}

#ifdef CONFIG_NO_HZ_COMMON
/*
 * NOHZ idle load balancing (ILB) details:
 *
 * - When one of the busy CPUs notices that there may be an idle rebalancing
 *   needed, they will kick the idle load balancer, which then does idle
 *   load balancing for all the idle CPUs.
 */
/*
 * find_new_ilb - 从允许承担内核维护工作的空闲 CPU 中挑选 NO_HZ 均衡代理。
 */
static inline int find_new_ilb(void)
{
	const struct cpumask *hk_mask;
	int ilb_cpu;

	hk_mask = housekeeping_cpumask(HK_TYPE_KERNEL_NOISE);

	for_each_cpu_and(ilb_cpu, nohz.idle_cpus_mask, hk_mask) {

		if (ilb_cpu == smp_processor_id())
			continue;

		if (idle_cpu(ilb_cpu))
			return ilb_cpu;
	}

	return -1;
}

/*
 * Kick a CPU to do the NOHZ balancing, if it is time for it, via a cross-CPU
 * SMP function call (IPI).
 *
 * We pick the first idle CPU in the HK_TYPE_KERNEL_NOISE housekeeping set
 * (if there is one).
 */
/*
 * kick_ilb - 合并 NO_HZ 维护请求，并在必要时异步通知选中的空闲代理 CPU。
 */
static void kick_ilb(unsigned int flags)
{
	int ilb_cpu;

	/*
	 * Increase nohz.next_balance only when if full ilb is triggered but
	 * not if we only update stats.
	 */
	if (flags & NOHZ_BALANCE_KICK)
		nohz.next_balance = jiffies+1;

	ilb_cpu = find_new_ilb();
	if (ilb_cpu < 0)
		return;

	/*
	 * Don't bother if no new NOHZ balance work items for ilb_cpu,
	 * i.e. all bits in flags are already set in ilb_cpu.
	 */
	if ((atomic_read(nohz_flags(ilb_cpu)) & flags) == flags)
		return;

	/*
	 * Access to rq::nohz_csd is serialized by NOHZ_KICK_MASK; he who sets
	 * the first flag owns it; cleared by nohz_csd_func().
	 */
	flags = atomic_fetch_or(flags, nohz_flags(ilb_cpu));
	if (flags & NOHZ_KICK_MASK)
		return;

	/*
	 * This way we generate an IPI on the target CPU which
	 * is idle, and the softirq performing NOHZ idle load balancing
	 * will be run before returning from the IPI.
	 */
	smp_call_function_single_async(ilb_cpu, &cpu_rq(ilb_cpu)->nohz_csd);
}

/*
 * Current decision point for kicking the idle load balancer in the presence
 * of idle CPUs in the system.
 */
/*
 * nohz_balancer_kick - 忙 CPU 根据排队、容量和历史负载状态，
 * 请求空闲 CPU 代做维护或均衡。
 */
static void nohz_balancer_kick(struct rq *rq)
{
	unsigned long now = jiffies;
	struct sched_domain_shared *sds;
	struct sched_domain *sd;
	int nr_busy, i, cpu = rq->cpu;
	unsigned int flags = 0;

	if (unlikely(rq->idle_balance))
		return;

	/*
	 * We may be recently in ticked or tickless idle mode. At the first
	 * busy tick after returning from idle, we will update the busy stats.
	 */
	nohz_balance_exit_idle(rq);

	/*
	 * None are in tickless mode and hence no need for NOHZ idle load
	 * balancing:
	 */
	if (likely(!atomic_read(&nohz.nr_cpus)))
		return;

	if (READ_ONCE(nohz.has_blocked) &&
	    time_after(now, READ_ONCE(nohz.next_blocked)))
		flags = NOHZ_STATS_KICK;

	if (time_before(now, nohz.next_balance))
		goto out;

	if (rq->nr_running >= 2) {
		flags = NOHZ_STATS_KICK | NOHZ_BALANCE_KICK;
		goto out;
	}

	rcu_read_lock();

	sd = rcu_dereference(rq->sd);
	if (sd) {
		/*
		 * If there's a runnable CFS task and the current CPU has reduced
		 * capacity, kick the ILB to see if there's a better CPU to run on:
		 */
		if (rq->cfs.h_nr_runnable >= 1 && check_cpu_capacity(rq, sd)) {
			flags = NOHZ_STATS_KICK | NOHZ_BALANCE_KICK;
			goto unlock;
		}
	}

	sd = rcu_dereference(per_cpu(sd_asym_packing, cpu));
	if (sd) {
		/*
		 * When ASYM_PACKING; see if there's a more preferred CPU
		 * currently idle; in which case, kick the ILB to move tasks
		 * around.
		 *
		 * When balancing between cores, all the SMT siblings of the
		 * preferred CPU must be idle.
		 */
		for_each_cpu_and(i, sched_domain_span(sd), nohz.idle_cpus_mask) {
			if (sched_asym(sd, i, cpu)) {
				flags = NOHZ_STATS_KICK | NOHZ_BALANCE_KICK;
				goto unlock;
			}
		}
	}

	sd = rcu_dereference(per_cpu(sd_asym_cpucapacity, cpu));
	if (sd) {
		/*
		 * When ASYM_CPUCAPACITY; see if there's a higher capacity CPU
		 * to run the misfit task on.
		 */
		if (check_misfit_status(rq)) {
			flags = NOHZ_STATS_KICK | NOHZ_BALANCE_KICK;
			goto unlock;
		}

		/*
		 * For asymmetric systems, we do not want to nicely balance
		 * cache use, instead we want to embrace asymmetry and only
		 * ensure tasks have enough CPU capacity.
		 *
		 * Skip the LLC logic because it's not relevant in that case.
		 */
		goto unlock;
	}

	sds = rcu_dereference(per_cpu(sd_llc_shared, cpu));
	if (sds) {
		/*
		 * If there is an imbalance between LLC domains (IOW we could
		 * increase the overall cache utilization), we need a less-loaded LLC
		 * domain to pull some load from. Likewise, we may need to spread
		 * load within the current LLC domain (e.g. packed SMT cores but
		 * other CPUs are idle). We can't really know from here how busy
		 * the others are - so just get a NOHZ balance going if it looks
		 * like this LLC domain has tasks we could move.
		 */
		nr_busy = atomic_read(&sds->nr_busy_cpus);
		if (nr_busy > 1) {
			flags = NOHZ_STATS_KICK | NOHZ_BALANCE_KICK;
			goto unlock;
		}
	}
unlock:
	rcu_read_unlock();
out:
	if (READ_ONCE(nohz.needs_update))
		flags |= NOHZ_NEXT_KICK;

	if (flags)
		kick_ilb(flags);
}

/*
 * set_cpu_sd_state_busy - 将共享缓存域中的本 CPU 标记为忙，
 * 并更新共享忙 CPU 计数。
 */
static void set_cpu_sd_state_busy(int cpu)
{
	struct sched_domain *sd;

	rcu_read_lock();
	sd = rcu_dereference(per_cpu(sd_llc, cpu));

	if (!sd || !sd->nohz_idle)
		goto unlock;
	sd->nohz_idle = 0;

	atomic_inc(&sd->shared->nr_busy_cpus);
unlock:
	rcu_read_unlock();
}

/*
 * nohz_balance_exit_idle - 退出停 tick 空闲集合，
 * 撤销 NO_HZ 掩码和相关空闲计数。
 */
void nohz_balance_exit_idle(struct rq *rq)
{
	WARN_ON_ONCE(rq != this_rq());

	if (likely(!rq->nohz_tick_stopped))
		return;

	rq->nohz_tick_stopped = 0;
	cpumask_clear_cpu(rq->cpu, nohz.idle_cpus_mask);
	atomic_dec(&nohz.nr_cpus);

	set_cpu_sd_state_busy(rq->cpu);
}

/*
 * set_cpu_sd_state_idle - 将共享缓存域中的本 CPU 标记为空闲，
 * 并减少共享忙 CPU 计数。
 */
static void set_cpu_sd_state_idle(int cpu)
{
	struct sched_domain *sd;

	rcu_read_lock();
	sd = rcu_dereference(per_cpu(sd_llc, cpu));

	if (!sd || sd->nohz_idle)
		goto unlock;
	sd->nohz_idle = 1;

	atomic_dec(&sd->shared->nr_busy_cpus);
unlock:
	rcu_read_unlock();
}

/*
 * This routine will record that the CPU is going idle with tick stopped.
 * This info will be used in performing idle load balancing in the future.
 */
/*
 * nohz_balance_enter_idle - 登记停 tick 的空闲 CPU，
 * 并标记仍需处理的历史负载和均衡时间。
 */
void nohz_balance_enter_idle(int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	WARN_ON_ONCE(cpu != smp_processor_id());

	/* If this CPU is going down, then nothing needs to be done: */
	if (!cpu_active(cpu))
		return;

	/*
	 * Can be set safely without rq->lock held
	 * If a clear happens, it will have evaluated last additions because
	 * rq->lock is held during the check and the clear
	 */
	rq->has_blocked_load = 1;

	/*
	 * The tick is still stopped but load could have been added in the
	 * meantime. We set the nohz.has_blocked flag to trig a check of the
	 * *_avg. The CPU is already part of nohz.idle_cpus_mask so the clear
	 * of nohz.has_blocked can only happen after checking the new load
	 */
	if (rq->nohz_tick_stopped)
		goto out;

	/* If we're a completely isolated CPU, we don't play: */
	if (on_null_domain(rq))
		return;

	rq->nohz_tick_stopped = 1;

	cpumask_set_cpu(cpu, nohz.idle_cpus_mask);
	atomic_inc(&nohz.nr_cpus);

	/*
	 * Ensures that if nohz_idle_balance() fails to observe our
	 * @idle_cpus_mask store, it must observe the @has_blocked
	 * and @needs_update stores.
	 */
	smp_mb__after_atomic();

	set_cpu_sd_state_idle(cpu);

	WRITE_ONCE(nohz.needs_update, 1);
out:
	/*
	 * Each time a cpu enter idle, we assume that it has blocked load and
	 * enable the periodic update of the load of idle CPUs
	 */
	WRITE_ONCE(nohz.has_blocked, 1);
}

/*
 * update_nohz_stats - 对仍停 tick 且存在历史贡献的 CPU 执行一次到期负载衰减。
 */
static bool update_nohz_stats(struct rq *rq)
{
	unsigned int cpu = rq->cpu;

	if (!rq->has_blocked_load)
		return false;

	if (!cpumask_test_cpu(cpu, nohz.idle_cpus_mask))
		return false;

	if (!time_after(jiffies, READ_ONCE(rq->last_blocked_load_update_tick)))
		return true;

	sched_balance_update_blocked_averages(cpu);

	return rq->has_blocked_load;
}

/*
 * Internal function that runs load balance for all idle CPUs. The load balance
 * can be a simple update of blocked load or a complete load balance with
 * tasks movement depending of flags.
 */
/*
 * _nohz_idle_balance - 代理遍历停 tick 的空闲 CPU，
 * 按请求更新统计、拉取任务或汇总下次时刻。
 */
static void _nohz_idle_balance(struct rq *this_rq, unsigned int flags)
{
	/* Earliest time when we have to do rebalance again */
	unsigned long now = jiffies;
	unsigned long next_balance = now + 60*HZ;
	bool has_blocked_load = false;
	int update_next_balance = 0;
	int this_cpu = this_rq->cpu;
	int balance_cpu;
	struct rq *rq;

	WARN_ON_ONCE((flags & NOHZ_KICK_MASK) == NOHZ_BALANCE_KICK);

	/*
	 * We assume there will be no idle load after this update and clear
	 * the has_blocked flag. If a cpu enters idle in the mean time, it will
	 * set the has_blocked flag and trigger another update of idle load.
	 * Because a cpu that becomes idle, is added to idle_cpus_mask before
	 * setting the flag, we are sure to not clear the state and not
	 * check the load of an idle cpu.
	 *
	 * Same applies to idle_cpus_mask vs needs_update.
	 */
	if (flags & NOHZ_STATS_KICK)
		WRITE_ONCE(nohz.has_blocked, 0);
	if (flags & NOHZ_NEXT_KICK)
		WRITE_ONCE(nohz.needs_update, 0);

	/*
	 * Ensures that if we miss the CPU, we must see the has_blocked
	 * store from nohz_balance_enter_idle().
	 */
	smp_mb();

	/*
	 * Start with the next CPU after this_cpu so we will end with this_cpu and let a
	 * chance for other idle cpu to pull load.
	 */
	for_each_cpu_wrap(balance_cpu,  nohz.idle_cpus_mask, this_cpu+1) {
		if (!idle_cpu(balance_cpu))
			continue;

		/*
		 * If this CPU gets work to do, stop the load balancing
		 * work being done for other CPUs. Next load
		 * balancing owner will pick it up.
		 */
		if (!idle_cpu(this_cpu) && need_resched()) {
			if (flags & NOHZ_STATS_KICK)
				has_blocked_load = true;
			if (flags & NOHZ_NEXT_KICK)
				WRITE_ONCE(nohz.needs_update, 1);
			goto abort;
		}

		rq = cpu_rq(balance_cpu);

		if (flags & NOHZ_STATS_KICK)
			has_blocked_load |= update_nohz_stats(rq);

		/*
		 * If time for next balance is due,
		 * do the balance.
		 */
		if (time_after_eq(jiffies, rq->next_balance)) {
			struct rq_flags rf;

			rq_lock_irqsave(rq, &rf);
			update_rq_clock(rq);
			rq_unlock_irqrestore(rq, &rf);

			if (flags & NOHZ_BALANCE_KICK)
				sched_balance_domains(rq, CPU_IDLE);
		}

		if (time_after(next_balance, rq->next_balance)) {
			next_balance = rq->next_balance;
			update_next_balance = 1;
		}
	}

	/*
	 * next_balance will be updated only when there is a need.
	 * When the CPU is attached to null domain for ex, it will not be
	 * updated.
	 */
	if (likely(update_next_balance))
		nohz.next_balance = next_balance;

	if (flags & NOHZ_STATS_KICK)
		WRITE_ONCE(nohz.next_blocked,
			   now + msecs_to_jiffies(LOAD_AVG_PERIOD));

abort:
	/* There is still blocked load, enable periodic update */
	if (has_blocked_load)
		WRITE_ONCE(nohz.has_blocked, 1);
}

/*
 * In CONFIG_NO_HZ_COMMON case, the idle balance kickee will do the
 * rebalancing for all the CPUs for whom scheduler ticks are stopped.
 */
/*
 * nohz_idle_balance - 消费本 CPU 收到的 NO_HZ 请求；仍为空闲时执行代理维护。
 */
static bool nohz_idle_balance(struct rq *this_rq, enum cpu_idle_type idle)
{
	unsigned int flags = this_rq->nohz_idle_balance;

	if (!flags)
		return false;

	this_rq->nohz_idle_balance = 0;

	if (idle != CPU_IDLE)
		return false;

	_nohz_idle_balance(this_rq, flags);

	return true;
}

/*
 * Check if we need to directly run the ILB for updating blocked load before
 * entering idle state. Here we run ILB directly without issuing IPIs.
 *
 * Note that when this function is called, the tick may not yet be stopped on
 * this CPU yet. nohz.idle_cpus_mask is updated only when tick is stopped and
 * cleared on the next busy tick. In other words, nohz.idle_cpus_mask updates
 * don't align with CPUs enter/exit idle to avoid bottlenecks due to high idle
 * entry/exit rate (usec). So it is possible that _nohz_idle_balance() is
 * called from this function on (this) CPU that's not yet in the mask. That's
 * OK because the goal of nohz_run_idle_balance() is to run ILB only for
 * updating the blocked load of already idle CPUs without waking up one of
 * those idle CPUs and outside the preempt disable / IRQ off phase of the local
 * cpu about to enter idle, because it can take a long time.
 */
/*
 * nohz_run_idle_balance - 在进入空闲路径时处理仅更新历史负载的轻量请求。
 */
void nohz_run_idle_balance(int cpu)
{
	unsigned int flags;

	flags = atomic_fetch_andnot(NOHZ_NEWILB_KICK, nohz_flags(cpu));

	/*
	 * Update the blocked load only if no SCHED_SOFTIRQ is about to happen
	 * (i.e. NOHZ_STATS_KICK set) and will do the same.
	 */
	if ((flags == NOHZ_NEWILB_KICK) && !need_resched())
		_nohz_idle_balance(cpu_rq(cpu), NOHZ_STATS_KICK);
}

/*
 * nohz_newidle_balance - 预计空闲时间足够且历史负载更新到期时，
 * 设置轻量维护请求。
 */
static void nohz_newidle_balance(struct rq *this_rq)
{
	int this_cpu = this_rq->cpu;

	/* Will wake up very soon. No time for doing anything else*/
	if (this_rq->avg_idle < sysctl_sched_migration_cost)
		return;

	/* Don't need to update blocked load of idle CPUs*/
	if (!READ_ONCE(nohz.has_blocked) ||
	    time_before(jiffies, READ_ONCE(nohz.next_blocked)))
		return;

	/*
	 * Set the need to trigger ILB in order to update blocked load
	 * before entering idle state.
	 */
	atomic_or(NOHZ_NEWILB_KICK, nohz_flags(this_cpu));
}

#else /* !CONFIG_NO_HZ_COMMON: */
/*
 * nohz_balancer_kick - 未启用 NO_HZ 公共支持时，不发送空闲代理请求。
 */
static inline void nohz_balancer_kick(struct rq *rq) { }

/*
 * nohz_idle_balance - 未启用 NO_HZ 公共支持时，无空闲代理工作可处理。
 */
static inline bool nohz_idle_balance(struct rq *this_rq, enum cpu_idle_type idle)
{
	return false;
}

/*
 * nohz_newidle_balance - 未启用 NO_HZ 公共支持时，不安排停 tick 负载维护。
 */
static inline void nohz_newidle_balance(struct rq *this_rq) { }
#endif /* !CONFIG_NO_HZ_COMMON */

/*
 * sched_balance_newidle is called by schedule() if this_cpu is about to become
 * idle. Attempts to pull tasks from other CPUs.
 *
 * Returns:
 *   < 0 - we released the lock and there are !fair tasks present
 *     0 - failed, no new tasks
 *   > 0 - success, new (fair) tasks present
 */
/*
 * sched_balance_newidle - CPU 即将空闲时，
 * 按预期空闲时长和扫描成本尝试从其他 CPU 拉取任务。
 * 返回正数表示有公平任务，0 表示仍无任务，
 * 负数表示应重新选择更高优先级类。
 */
static int sched_balance_newidle(struct rq *this_rq, struct rq_flags *rf)
{
	unsigned long next_balance = jiffies + HZ;
	int this_cpu = this_rq->cpu;
	int continue_balancing = 1;
	u64 t0, t1, curr_cost = 0;
	struct sched_domain *sd;
	int pulled_task = 0;

	update_misfit_status(NULL, this_rq);

	/*
	 * There is a task waiting to run. No need to search for one.
	 * Return 0; the task will be enqueued when switching to idle.
	 */
	if (this_rq->ttwu_pending)
		return 0;

	/*
	 * We must set idle_stamp _before_ calling sched_balance_rq()
	 * for CPU_NEWLY_IDLE, such that we measure the this duration
	 * as idle time.
	 */
	this_rq->idle_stamp = rq_clock(this_rq);

	/*
	 * Do not pull tasks towards !active CPUs...
	 */
	if (!cpu_active(this_cpu))
		return 0;

	/*
	 * This is OK, because current is on_cpu, which avoids it being picked
	 * for load-balance and preemption/IRQs are still disabled avoiding
	 * further scheduler activity on it and we're being very careful to
	 * re-start the picking loop.
	 */
	rq_unpin_lock(this_rq, rf);

	rcu_read_lock();
	sd = rcu_dereference_check_sched_domain(this_rq->sd);
	if (!sd) {
		rcu_read_unlock();
		goto out;
	}

	if (!get_rd_overloaded(this_rq->rd) ||
	    this_rq->avg_idle < sd->max_newidle_lb_cost) {

		update_next_balance(sd, &next_balance);
		rcu_read_unlock();
		goto out;
	}
	rcu_read_unlock();

	/*
	 * 跨 CPU 均衡前释放本地 rq 锁，
	 * 返回后必须重新确认是否已有任务被唤醒到本 CPU。
	 */
	raw_spin_rq_unlock(this_rq);

	t0 = sched_clock_cpu(this_cpu);
	sched_balance_update_blocked_averages(this_cpu);

	rcu_read_lock();
	for_each_domain(this_cpu, sd) {
		u64 domain_cost;

		update_next_balance(sd, &next_balance);

		if (this_rq->avg_idle < curr_cost + sd->max_newidle_lb_cost)
			break;

		if (sd->flags & SD_BALANCE_NEWIDLE) {
			unsigned int weight = 1;

			if (sched_feat(NI_RANDOM)) {
				/*
				 * Throw a 1k sided dice; and only run
				 * newidle_balance according to the success
				 * rate.
				 */
				u32 d1k = sched_rng() % 1024;
				weight = 1 + sd->newidle_ratio;
				if (d1k > weight) {
					update_newidle_stats(sd, 0);
					continue;
				}
				weight = (1024 + weight/2) / weight;
			}

			pulled_task = sched_balance_rq(this_cpu, this_rq,
						   sd, CPU_NEWLY_IDLE,
						   &continue_balancing);

			t1 = sched_clock_cpu(this_cpu);
			domain_cost = t1 - t0;
			curr_cost += domain_cost;
			t0 = t1;

			/*
			 * Track max cost of a domain to make sure to not delay the
			 * next wakeup on the CPU.
			 */
			update_newidle_cost(sd, domain_cost, weight * !!pulled_task);
		}

		/*
		 * Stop searching for tasks to pull if there are
		 * now runnable tasks on this rq.
		 */
		if (pulled_task || !continue_balancing)
			break;
	}
	rcu_read_unlock();

	raw_spin_rq_lock(this_rq);

	if (curr_cost > this_rq->max_idle_balance_cost)
		this_rq->max_idle_balance_cost = curr_cost;

	/*
	 * While browsing the domains, we released the rq lock, a task could
	 * have been enqueued in the meantime. Since we're not going idle,
	 * pretend we pulled a task.
	 */
	if (this_rq->cfs.h_nr_queued && !pulled_task)
		pulled_task = 1;

	/* Is there a task of a high priority class? */
	/*
	 * 出现非公平类任务时，
	 * 返回负值使外层重新执行调度类优先级选择。
	 */
	if (this_rq->nr_running != this_rq->cfs.h_nr_queued)
		pulled_task = -1;

out:
	/* Move the next balance forward */
	if (time_after(this_rq->next_balance, next_balance))
		this_rq->next_balance = next_balance;

	if (pulled_task)
		this_rq->idle_stamp = 0;
	else
		nohz_newidle_balance(this_rq);

	rq_repin_lock(this_rq, rf);

	return pulled_task;
}

/*
 * This softirq handler is triggered via SCHED_SOFTIRQ from two places:
 *
 * - directly from the local sched_tick() for periodic load balancing
 *
 * - indirectly from a remote sched_tick() for NOHZ idle balancing
 *   through the SMP cross-call nohz_csd_func()
 */
/*
 * sched_balance_softirq - 处理调度软中断，优先完成 NO_HZ 代理请求，
 * 否则执行本地周期均衡。
 */
static __latent_entropy void sched_balance_softirq(void)
{
	struct rq *this_rq = this_rq();
	enum cpu_idle_type idle = this_rq->idle_balance;
	/*
	 * If this CPU has a pending NOHZ_BALANCE_KICK, then do the
	 * balancing on behalf of the other idle CPUs whose ticks are
	 * stopped. Do nohz_idle_balance *before* sched_balance_domains to
	 * give the idle CPUs a chance to load balance. Else we may
	 * load balance only within the local sched_domain hierarchy
	 * and abort nohz_idle_balance altogether if we pull some load.
	 */
	if (nohz_idle_balance(this_rq, idle))
		return;

	/* normal load balance */
	sched_balance_update_blocked_averages(this_rq->cpu);
	sched_balance_domains(this_rq, idle);
}

/*
 * Trigger the SCHED_SOFTIRQ if it is time to do periodic load balancing.
 */
/*
 * sched_balance_trigger - tick 路径发现均衡到期时触发调度软中断，
 * 并按需唤起 NO_HZ 代理。
 */
void sched_balance_trigger(struct rq *rq)
{
	/*
	 * Don't need to rebalance while attached to NULL domain or
	 * runqueue CPU is not active
	 */
	if (unlikely(on_null_domain(rq) || !cpu_active(cpu_of(rq))))
		return;

	if (time_after_eq(jiffies, rq->next_balance))
		raise_softirq(SCHED_SOFTIRQ);

	nohz_balancer_kick(rq);
}

/*
 * rq_online_fair - CPU 上线时刷新时间片缩放和各组本地带宽控制状态。
 */
static void rq_online_fair(struct rq *rq)
{
	update_sysctl();

	update_runtime_enabled(rq);
}

/*
 * rq_offline_fair - CPU 离线时刷新缩放、解除本地限流并清除组负载贡献。
 */
static void rq_offline_fair(struct rq *rq)
{
	update_sysctl();

	/* Ensure any throttled groups are reachable by pick_next_task */
	unthrottle_offline_cfs_rqs(rq);

	/* Ensure that we remove rq contribution to group share: */
	clear_tg_offline_cfs_rqs(rq);
}

#ifdef CONFIG_SCHED_CORE
/*
 * __entity_slice_used - 按假定的最少竞争任务数，
 * 判断当前实体是否已使用足够时间片。
 */
static inline bool
__entity_slice_used(struct sched_entity *se, int min_nr_tasks)
{
	u64 rtime = se->sum_exec_runtime - se->prev_sum_exec_runtime;
	u64 slice = se->slice;

	return (rtime * min_nr_tasks > slice);
}

#define MIN_NR_TASKS_DURING_FORCEIDLE	2
/*
 * task_tick_core - 核心调度发生强制空闲时，适时触发重选，
 * 给受阻的兄弟线程任务机会。
 */
static inline void task_tick_core(struct rq *rq, struct task_struct *curr)
{
	if (!sched_core_enabled(rq))
		return;

	/*
	 * If runqueue has only one task which used up its slice and
	 * if the sibling is forced idle, then trigger schedule to
	 * give forced idle task a chance.
	 *
	 * sched_slice() considers only this active rq and it gets the
	 * whole slice. But during force idle, we have siblings acting
	 * like a single runqueue and hence we need to consider runnable
	 * tasks on this CPU and the forced idle CPU. Ideally, we should
	 * go through the forced idle rq, but that would be a perf hit.
	 * We can assume that the forced idle CPU has at least
	 * MIN_NR_TASKS_DURING_FORCEIDLE - 1 tasks and use that to check
	 * if we need to give up the CPU.
	 */
	if (rq->core->core_forceidle_count && rq->cfs.nr_queued == 1 &&
	    __entity_slice_used(&curr->se, MIN_NR_TASKS_DURING_FORCEIDLE))
		resched_curr(rq);
}

/*
 * Consider any infeasible weight scenario. Take for instance two tasks,
 * each bound to their respective sibling, one with weight 1 and one with
 * weight 2. Then the lower weight task will run ahead of the higher weight
 * task without bound.
 *
 * This utterly destroys the concept of a shared time base.
 *
 * Remember; all this is about a proportionally fair scheduling, where each
 * tasks receives:
 *
 *              w_i
 *   dt_i = ---------- dt                                     (1)
 *          \Sum_j w_j
 *
 * which we do by tracking a virtual time, s_i:
 *
 *          1
 *   s_i = --- d[t]_i                                         (2)
 *         w_i
 *
 * Where d[t] is a delta of discrete time, while dt is an infinitesimal.
 * The immediate corollary is that the ideal schedule S, where (2) to use
 * an infinitesimal delta, is:
 *
 *           1
 *   S = ---------- dt                                        (3)
 *       \Sum_i w_i
 *
 * From which we can define the lag, or deviation from the ideal, as:
 *
 *   lag(i) = S - s_i                                         (4)
 *
 * And since the one and only purpose is to approximate S, we get that:
 *
 *   \Sum_i w_i lag(i) := 0                                   (5)
 *
 * If this were not so, we no longer converge to S, and we can no longer
 * claim our scheduler has any of the properties we derive from S. This is
 * exactly what you did above, you broke it!
 *
 *
 * Let's continue for a while though; to see if there is anything useful to
 * be learned. We can combine (1)-(3) or (4)-(5) and express S in s_i:
 *
 *       \Sum_i w_i s_i
 *   S = --------------                                       (6)
 *         \Sum_i w_i
 *
 * Which gives us a way to compute S, given our s_i. Now, if you've read
 * our code, you know that we do not in fact do this, the reason for this
 * is two-fold. Firstly, computing S in that way requires a 64bit division
 * for every time we'd use it (see 12), and secondly, this only describes
 * the steady-state, it doesn't handle dynamics.
 *
 * Anyway, in (6):  s_i -> x + (s_i - x), to get:
 *
 *           \Sum_i w_i (s_i - x)
 *   S - x = --------------------                             (7)
 *              \Sum_i w_i
 *
 * Which shows that S and s_i transform alike (which makes perfect sense
 * given that S is basically the (weighted) average of s_i).
 *
 * So the thing to remember is that the above is strictly UP. It is
 * possible to generalize to multiple runqueues -- however it gets really
 * yuck when you have to add affinity support, as illustrated by our very
 * first counter-example.
 *
 * Luckily I think we can avoid needing a full multi-queue variant for
 * core-scheduling (or load-balancing). The crucial observation is that we
 * only actually need this comparison in the presence of forced-idle; only
 * then do we need to tell if the stalled rq has higher priority over the
 * other.
 *
 * [XXX assumes SMT2; better consider the more general case, I suspect
 * it'll work out because our comparison is always between 2 rqs and the
 * answer is only interesting if one of them is forced-idle]
 *
 * And (under assumption of SMT2) when there is forced-idle, there is only
 * a single queue, so everything works like normal.
 *
 * Let, for our runqueue 'k':
 *
 *   T_k = \Sum_i w_i s_i
 *   W_k = \Sum_i w_i      ; for all i of k                  (8)
 *
 * Then we can write (6) like:
 *
 *         T_k
 *   S_k = ---                                               (9)
 *         W_k
 *
 * From which immediately follows that:
 *
 *           T_k + T_l
 *   S_k+l = ---------                                       (10)
 *           W_k + W_l
 *
 * On which we can define a combined lag:
 *
 *   lag_k+l(i) := S_k+l - s_i                               (11)
 *
 * And that gives us the tools to compare tasks across a combined runqueue.
 *
 *
 * Combined this gives the following:
 *
 *  a) when a runqueue enters force-idle, sync it against it's sibling rq(s)
 *     using (7); this only requires storing single 'time'-stamps.
 *
 *  b) when comparing tasks between 2 runqueues of which one is forced-idle,
 *     compare the combined lag, per (11).
 *
 * Now, of course cgroups (I so hate them) make this more interesting in
 * that a) seems to suggest we need to iterate all cgroup on a CPU at such
 * boundaries, but I think we can avoid that. The force-idle is for the
 * whole CPU, all it's rqs. So we can mark it in the root and lazily
 * propagate downward on demand.
 */

/*
 * So this sync is basically a relative reset of S to 0.
 *
 * So with 2 queues, when one goes idle, we drop them both to 0 and one
 * then increases due to not being idle, and the idle one builds up lag to
 * get re-elected. So far so simple, right?
 *
 * When there's 3, we can have the situation where 2 run and one is idle,
 * we sync to 0 and let the idle one build up lag to get re-election. Now
 * suppose another one also drops idle. At this point dropping all to 0
 * again would destroy the built-up lag from the queue that was already
 * idle, not good.
 *
 * So instead of syncing everything, we can:
 *
 *   less := !((s64)(s_a - s_b) <= 0)
 *
 *   (v_a - S_a) - (v_b - S_b) == v_a - v_b - S_a + S_b
 *                             == v_a - (v_b - S_a + S_b)
 *
 * IOW, we can recast the (lag) comparison to a one-sided difference.
 * So if then, instead of syncing the whole queue, sync the idle queue
 * against the active queue with S_a + S_b at the point where we sync.
 *
 * (XXX consider the implication of living in a cyclic group: N / 2^n N)
 *
 * This gives us means of syncing single queues against the active queue,
 * and for already idle queues to preserve their build-up lag.
 *
 * Of course, then we get the situation where there's 2 active and one
 * going idle, who do we pick to sync against? Theory would have us sync
 * against the combined S, but as we've already demonstrated, there is no
 * such thing in infeasible weight scenarios.
 *
 * One thing I've considered; and this is where that core_active rudiment
 * came from, is having active queues sync up between themselves after
 * every tick. This limits the observed divergence due to the work
 * conservancy.
 *
 * On top of that, we can improve upon things by employing (10) here.
 */

/*
 * se_fi_update - Update the cfs_rq->zero_vruntime_fi in a CFS hierarchy if needed.
 */
/*
 * se_fi_update - 沿组层级记录强制空闲比较使用的虚拟时间原点，
 * 同一序列避免重复更新。
 */
static void se_fi_update(const struct sched_entity *se, unsigned int fi_seq,
			 bool forceidle)
{
	for_each_sched_entity(se) {
		struct cfs_rq *cfs_rq = cfs_rq_of(se);

		if (forceidle) {
			if (cfs_rq->forceidle_seq == fi_seq)
				break;
			cfs_rq->forceidle_seq = fi_seq;
		}

		cfs_rq->zero_vruntime_fi = cfs_rq->zero_vruntime;
	}
}

/*
 * task_vruntime_update - 为公平任务更新核心调度所需的强制空闲时间基准。
 */
void task_vruntime_update(struct rq *rq, struct task_struct *p, bool in_fi)
{
	struct sched_entity *se = &p->se;

	if (p->sched_class != &fair_sched_class)
		return;

	se_fi_update(se, rq->core->core_forceidle_seq, in_fi);
}

/*
 * cfs_prio_less - 比较同一物理核心上两个公平任务的相对优先级。
 * 先对齐组层级，再按各队列的强制空闲基准归一化 vruntime；
 * a 较低时返回 true。
 */
bool cfs_prio_less(const struct task_struct *a, const struct task_struct *b,
			bool in_fi)
{
	struct rq *rq = task_rq(a);
	const struct sched_entity *sea = &a->se;
	const struct sched_entity *seb = &b->se;
	struct cfs_rq *cfs_rqa;
	struct cfs_rq *cfs_rqb;
	s64 delta;

	WARN_ON_ONCE(task_rq(b)->core != rq->core);

#ifdef CONFIG_FAIR_GROUP_SCHED
	/*
	 * Find an se in the hierarchy for tasks a and b, such that the se's
	 * are immediate siblings.
	 */
	while (sea->cfs_rq->tg != seb->cfs_rq->tg) {
		int sea_depth = sea->depth;
		int seb_depth = seb->depth;

		if (sea_depth >= seb_depth)
			sea = parent_entity(sea);
		if (sea_depth <= seb_depth)
			seb = parent_entity(seb);
	}

	se_fi_update(sea, rq->core->core_forceidle_seq, in_fi);
	se_fi_update(seb, rq->core->core_forceidle_seq, in_fi);

	cfs_rqa = sea->cfs_rq;
	cfs_rqb = seb->cfs_rq;
#else /* !CONFIG_FAIR_GROUP_SCHED: */
	cfs_rqa = &task_rq(a)->cfs;
	cfs_rqb = &task_rq(b)->cfs;
#endif /* !CONFIG_FAIR_GROUP_SCHED */

	/*
	 * Find delta after normalizing se's vruntime with its cfs_rq's
	 * zero_vruntime_fi, which would have been updated in prior calls
	 * to se_fi_update().
	 */
	/*
	 * 不同 CPU 的虚拟时间原点不同，
	 * 减去各自基准后才能作核心调度的相对比较。
	 */
	delta = vruntime_op(sea->vruntime, "-", seb->vruntime) +
		vruntime_op(cfs_rqb->zero_vruntime_fi, "-", cfs_rqa->zero_vruntime_fi);

	return delta > 0;
}

/*
 * task_is_throttled_fair：
 * 为核心调度检查任务所在组在指定 CPU 上的层级限流状态。
 */
static int task_is_throttled_fair(struct task_struct *p, int cpu)
{
	struct cfs_rq *cfs_rq;

#ifdef CONFIG_FAIR_GROUP_SCHED
	cfs_rq = task_group(p)->cfs_rq[cpu];
#else
	cfs_rq = &cpu_rq(cpu)->cfs;
#endif
	return throttled_hierarchy(cfs_rq);
}
#else /* !CONFIG_SCHED_CORE: */
/*
 * task_tick_core - 未启用核心调度时，
 * 无需处理 SMT 强制空闲带来的时间片竞争。
 */
static inline void task_tick_core(struct rq *rq, struct task_struct *curr) {}
#endif /* !CONFIG_SCHED_CORE */

/*
 * scheduler tick hitting a task of our scheduling class.
 *
 * NOTE: This function can be called remotely by the tick offload that
 * goes along full dynticks. Therefore no local assumption can be made
 * and everything must be accessed through the @rq and @curr passed in
 * parameters.
 */
/*
 * task_tick_fair - 公平类 tick 入口，逐层更新实体，
 * 再维护 NUMA、容量匹配及核心调度状态。
 * 可能由远端 tick 代执行，所有 CPU 状态须通过传入的 rq 和 curr 获取。
 */
static void task_tick_fair(struct rq *rq, struct task_struct *curr, int queued)
{
	struct cfs_rq *cfs_rq;
	struct sched_entity *se = &curr->se;

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);
		entity_tick(cfs_rq, se, queued);
	}

	if (static_branch_unlikely(&sched_numa_balancing))
		task_tick_numa(rq, curr);

	update_misfit_status(curr, rq);
	check_update_overutilized_status(task_rq(curr));

	task_tick_core(rq, curr);
}

/*
 * called on fork with the child task as argument from the parent's context
 *  - child not yet on the tasklist
 *  - preemption disabled
 */
/*
 * task_fork_fair - 在父任务上下文为子任务初始化亲和性允许的最大 CPU 容量。
 */
static void task_fork_fair(struct task_struct *p)
{
	set_task_max_allowed_capacity(p);
}

/*
 * Priority of the task has changed. Check to see if we preempt
 * the current task.
 */
/*
 * prio_changed_fair - 优先级改变后，
 * 按当前执行身份决定请求重调度或检查唤醒抢占。
 */
static void
prio_changed_fair(struct rq *rq, struct task_struct *p, int oldprio)
{
	if (!task_on_rq_queued(p))
		return;

	if (rq->cfs.nr_queued == 1)
		return;

	/*
	 * Reschedule if we are currently running on this runqueue and
	 * our priority decreased, or if we are not currently running on
	 * this runqueue and our priority is higher than the current's
	 */
	if (task_current_donor(rq, p)) {
		if (p->prio > oldprio)
			resched_curr(rq);
	} else
		wakeup_preempt(rq, p, 0);
}

#ifdef CONFIG_FAIR_GROUP_SCHED
/*
 * Propagate the changes of the sched_entity across the tg tree to make it
 * visible to the root
 */
/*
 * propagate_entity_cfs_rq - 把实体负载变化上传到根，
 * 并确保未暂停 PELT 的相关队列留在维护链表。
 */
static void propagate_entity_cfs_rq(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);

	/*
	 * If a task gets attached to this cfs_rq and before being queued,
	 * it gets migrated to another CPU due to reasons like affinity
	 * change, make sure this cfs_rq stays on leaf cfs_rq list to have
	 * that removed load decayed or it can cause faireness problem.
	 */
	if (!cfs_rq_pelt_clock_throttled(cfs_rq))
		list_add_leaf_cfs_rq(cfs_rq);

	/* Start to propagate at parent */
	se = se->parent;

	for_each_sched_entity(se) {
		cfs_rq = cfs_rq_of(se);

		update_load_avg(cfs_rq, se, UPDATE_TG);

		if (!cfs_rq_pelt_clock_throttled(cfs_rq))
			list_add_leaf_cfs_rq(cfs_rq);
	}

	assert_list_leaf_cfs_rq(rq_of(cfs_rq));
}
#else /* !CONFIG_FAIR_GROUP_SCHED: */
/*
 * propagate_entity_cfs_rq - 未启用组调度时，无祖先队列需要传播负载变化。
 */
static void propagate_entity_cfs_rq(struct sched_entity *se) { }
#endif /* !CONFIG_FAIR_GROUP_SCHED */

/*
 * detach_entity_cfs_rq - 同步并撤销实体在旧队列中的 PELT 贡献，
 * 再向组层级传播变化。
 */
static void detach_entity_cfs_rq(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);

	/*
	 * In case the task sched_avg hasn't been attached:
	 * - A forked task which hasn't been woken up by wake_up_new_task().
	 * - A task which has been woken up by try_to_wake_up() but is
	 *   waiting for actually being woken up by sched_ttwu_pending().
	 */
	if (!se->avg.last_update_time)
		return;

	/* Catch up with the cfs_rq and remove our load when we leave */
	update_load_avg(cfs_rq, se, 0);
	detach_entity_load_avg(cfs_rq, se);
	update_tg_load_avg(cfs_rq);
	propagate_entity_cfs_rq(se);
}

/*
 * attach_entity_cfs_rq - 将实体历史统计接入所属队列，
 * 并更新组总量及祖先队列。
 */
static void attach_entity_cfs_rq(struct sched_entity *se)
{
	struct cfs_rq *cfs_rq = cfs_rq_of(se);

	/* Synchronize entity with its cfs_rq */
	update_load_avg(cfs_rq, se, sched_feat(ATTACH_AGE_LOAD) ? 0 : SKIP_AGE_LOAD);
	attach_entity_load_avg(cfs_rq, se);
	update_tg_load_avg(cfs_rq);
	propagate_entity_cfs_rq(se);
}

/*
 * detach_task_cfs_rq - 以任务实体为入口，解除它与公平队列的负载统计关联。
 */
static void detach_task_cfs_rq(struct task_struct *p)
{
	struct sched_entity *se = &p->se;

	detach_entity_cfs_rq(se);
}

/*
 * attach_task_cfs_rq - 以任务实体为入口，建立它与公平队列的负载统计关联。
 */
static void attach_task_cfs_rq(struct task_struct *p)
{
	struct sched_entity *se = &p->se;

	attach_entity_cfs_rq(se);
}

/*
 * switched_from_fair - 任务离开公平调度类时，撤销其公平负载统计关联。
 */
static void switched_from_fair(struct rq *rq, struct task_struct *p)
{
	detach_task_cfs_rq(p);
}

/*
 * switched_to_fair - 任务切入公平类时接入统计并刷新容量信息，
 * 在队时重新评估调度顺序。
 */
static void switched_to_fair(struct rq *rq, struct task_struct *p)
{
	WARN_ON_ONCE(p->se.sched_delayed);

	attach_task_cfs_rq(p);

	set_task_max_allowed_capacity(p);

	if (task_on_rq_queued(p)) {
		/*
		 * We were most likely switched from sched_rt, so
		 * kick off the schedule if running, otherwise just see
		 * if we can still preempt the current task.
		 */
		if (task_current_donor(rq, p))
			resched_curr(rq);
		else
			wakeup_preempt(rq, p, 0);
	}
}

/*
 * __set_next_task_fair - 维护任务最近使用顺序，
 * 并在首次选中时更新高精度 tick、misfit 和 tick 依赖。
 */
static void __set_next_task_fair(struct rq *rq, struct task_struct *p, bool first)
{
	struct sched_entity *se = &p->se;

	if (task_on_rq_queued(p)) {
		/*
		 * Move the next running task to the front of the list, so our
		 * cfs_tasks list becomes MRU one.
		 */
		list_move(&se->group_node, &rq->cfs_tasks);
	}
	if (!first)
		return;

	WARN_ON_ONCE(se->sched_delayed);

	if (hrtick_enabled_fair(rq))
		hrtick_start_fair(rq, p);

	update_misfit_status(p, rq);
	sched_fair_update_stop_tick(rq, p);
}

/*
 * Account for a task changing its policy or group.
 *
 * This routine is mostly called to set cfs_rq->curr field when a task
 * migrates between groups/classes.
 */
/*
 * set_next_task_fair - 在任务成为公平类当前任务时设置各层 curr，
 * 并保证新队列已检查带宽预算。
 */
static void set_next_task_fair(struct rq *rq, struct task_struct *p, bool first)
{
	struct sched_entity *se = &p->se;

	for_each_sched_entity(se) {
		struct cfs_rq *cfs_rq = cfs_rq_of(se);

		set_next_entity(cfs_rq, se, first);
		/* ensure bandwidth has been allocated on our new cfs_rq */
		account_cfs_rq_runtime(cfs_rq, 0);
	}

	__set_next_task_fair(rq, p, first);
}

/*
 * init_cfs_rq - 初始化公平队列红黑树、虚拟时间原点及异步移除统计锁。
 */
void init_cfs_rq(struct cfs_rq *cfs_rq)
{
	cfs_rq->tasks_timeline = RB_ROOT_CACHED;
	cfs_rq->zero_vruntime = (u64)(-(1LL << 20));
	raw_spin_lock_init(&cfs_rq->removed.lock);
}

#ifdef CONFIG_FAIR_GROUP_SCHED
/*
 * task_change_group_fair - 任务变更 cgroup 后，将其历史负载从旧组迁到新组；
 * 未首次唤醒任务暂跳过。
 */
static void task_change_group_fair(struct task_struct *p)
{
	/*
	 * We couldn't detach or attach a forked task which
	 * hasn't been woken up by wake_up_new_task().
	 */
	if (READ_ONCE(p->__state) == TASK_NEW)
		return;

	detach_task_cfs_rq(p);

	/* Tell se's cfs_rq has been changed -- migrated */
	p->se.avg.last_update_time = 0;
	set_task_rq(p, task_cpu(p));
	attach_task_cfs_rq(p);
}

/*
 * free_fair_sched_group - 释放公平任务组的各 CPU 队列、组实体以及索引数组。
 */
void free_fair_sched_group(struct task_group *tg)
{
	int i;

	for_each_possible_cpu(i) {
		if (tg->cfs_rq)
			kfree(tg->cfs_rq[i]);
		if (tg->se)
			kfree(tg->se[i]);
	}

	kfree(tg->cfs_rq);
	kfree(tg->se);
}

/*
 * alloc_fair_sched_group - 分配任务组的每 CPU 公平队列和调度实体，
 * 并建立父子层级关系。
 * 成功返回 1，分配失败返回 0，已保存到组中的资源由外层清理路径回收。
 */
int alloc_fair_sched_group(struct task_group *tg, struct task_group *parent)
{
	struct sched_entity *se;
	struct cfs_rq *cfs_rq;
	int i;

	tg->cfs_rq = kcalloc(nr_cpu_ids, sizeof(cfs_rq), GFP_KERNEL);
	if (!tg->cfs_rq)
		goto err;
	tg->se = kcalloc(nr_cpu_ids, sizeof(se), GFP_KERNEL);
	if (!tg->se)
		goto err;

	tg->shares = NICE_0_LOAD;

	init_cfs_bandwidth(tg_cfs_bandwidth(tg), tg_cfs_bandwidth(parent));

	for_each_possible_cpu(i) {
		cfs_rq = kzalloc_node(sizeof(struct cfs_rq),
				      GFP_KERNEL, cpu_to_node(i));
		if (!cfs_rq)
			goto err;

		se = kzalloc_node(sizeof(struct sched_entity_stats),
				  GFP_KERNEL, cpu_to_node(i));
		if (!se)
			goto err_free_rq;

		init_cfs_rq(cfs_rq);
		init_tg_cfs_entry(tg, cfs_rq, se, i, parent->se[i]);
		init_entity_runnable_average(se);
	}

	return 1;

err_free_rq:
	kfree(cfs_rq);
err:
	return 0;
}

/*
 * online_fair_sched_group - 逐 CPU 接入新组实体的负载统计，
 * 并继承当前父组限流状态。
 */
void online_fair_sched_group(struct task_group *tg)
{
	struct sched_entity *se;
	struct rq_flags rf;
	struct rq *rq;
	int i;

	for_each_possible_cpu(i) {
		rq = cpu_rq(i);
		se = tg->se[i];
		rq_lock_irq(rq, &rf);
		update_rq_clock(rq);
		attach_entity_cfs_rq(se);
		sync_throttle(tg, i);
		rq_unlock_irq(rq, &rf);
	}
}

/*
 * unregister_fair_sched_group - 注销空任务组的带宽和队列关联，
 * 处理残留的延迟出队及历史负载。
 */
void unregister_fair_sched_group(struct task_group *tg)
{
	int cpu;

	destroy_cfs_bandwidth(tg_cfs_bandwidth(tg));

	for_each_possible_cpu(cpu) {
		struct cfs_rq *cfs_rq = tg->cfs_rq[cpu];
		struct sched_entity *se = tg->se[cpu];
		struct rq *rq = cpu_rq(cpu);

		if (se) {
			if (se->sched_delayed) {
				guard(rq_lock_irqsave)(rq);
				if (se->sched_delayed) {
					update_rq_clock(rq);
					dequeue_entities(rq, se, DEQUEUE_SLEEP | DEQUEUE_DELAYED);
				}
				list_del_leaf_cfs_rq(cfs_rq);
			}
			remove_entity_load_avg(se);
		}

		/*
		 * Only empty task groups can be destroyed; so we can speculatively
		 * check on_list without danger of it being re-added.
		 */
		if (cfs_rq->on_list) {
			guard(rq_lock_irqsave)(rq);
			list_del_leaf_cfs_rq(cfs_rq);
		}
	}
}

/*
 * init_tg_cfs_entry - 连接任务组、每 CPU 子队列和父队列中的组实体。
 * se->cfs_rq 指向竞争所在的父队列，se->my_q 指向它代表的子队列。
 */
void init_tg_cfs_entry(struct task_group *tg, struct cfs_rq *cfs_rq,
			struct sched_entity *se, int cpu,
			struct sched_entity *parent)
{
	struct rq *rq = cpu_rq(cpu);

	cfs_rq->tg = tg;
	cfs_rq->rq = rq;
	init_cfs_rq_runtime(cfs_rq);

	tg->cfs_rq[cpu] = cfs_rq;
	tg->se[cpu] = se;

	/* se could be NULL for root_task_group */
	if (!se)
		return;

	if (!parent) {
		se->cfs_rq = &rq->cfs;
		se->depth = 0;
	} else {
		se->cfs_rq = parent->my_q;
		se->depth = parent->depth + 1;
	}

	/*
	 * 组实体在父队列竞争，选中后通过 my_q 向下进入本组的子队列。
	 */
	se->my_q = cfs_rq;
	/* guarantee group entities always have weight */
	update_load_set(&se->load, NICE_0_LOAD);
	se->parent = parent;
}

static DEFINE_MUTEX(shares_mutex);

/*
 * __sched_group_set_shares - 持有 shares_mutex 时更新组份额，
 * 并逐 CPU 重算沿途组实体权重。
 */
static int __sched_group_set_shares(struct task_group *tg, unsigned long shares)
{
	int i;

	lockdep_assert_held(&shares_mutex);

	/*
	 * We can't change the weight of the root cgroup.
	 */
	if (!tg->se[0])
		return -EINVAL;

	shares = clamp(shares, scale_load(MIN_SHARES), scale_load(MAX_SHARES));

	if (tg->shares == shares)
		return 0;

	tg->shares = shares;
	for_each_possible_cpu(i) {
		struct rq *rq = cpu_rq(i);
		struct sched_entity *se = tg->se[i];
		struct rq_flags rf;

		/* Propagate contribution to hierarchy */
		rq_lock_irqsave(rq, &rf);
		update_rq_clock(rq);
		for_each_sched_entity(se) {
			update_load_avg(cfs_rq_of(se), se, UPDATE_TG);
			update_cfs_group(se);
		}
		rq_unlock_irqrestore(rq, &rf);
	}

	return 0;
}

/*
 * sched_group_set_shares - 串行化组权重修改；空闲调度组使用固定低权重，
 * 拒绝直接修改份额。
 */
int sched_group_set_shares(struct task_group *tg, unsigned long shares)
{
	int ret;

	mutex_lock(&shares_mutex);
	if (tg_is_idle(tg))
		ret = -EINVAL;
	else
		ret = __sched_group_set_shares(tg, shares);
	mutex_unlock(&shares_mutex);

	return ret;
}

/*
 * sched_group_set_idle - 切换任务组的空闲调度属性，
 * 传播层级 idle 计数并设置相应默认权重。
 */
int sched_group_set_idle(struct task_group *tg, long idle)
{
	int i;

	if (tg == &root_task_group)
		return -EINVAL;

	if (idle < 0 || idle > 1)
		return -EINVAL;

	mutex_lock(&shares_mutex);

	if (tg->idle == idle) {
		mutex_unlock(&shares_mutex);
		return 0;
	}

	tg->idle = idle;

	for_each_possible_cpu(i) {
		struct rq *rq = cpu_rq(i);
		struct sched_entity *se = tg->se[i];
		struct cfs_rq *grp_cfs_rq = tg->cfs_rq[i];
		bool was_idle = cfs_rq_is_idle(grp_cfs_rq);
		long idle_task_delta;
		struct rq_flags rf;

		rq_lock_irqsave(rq, &rf);

		grp_cfs_rq->idle = idle;
		if (WARN_ON_ONCE(was_idle == cfs_rq_is_idle(grp_cfs_rq)))
			goto next_cpu;

		idle_task_delta = grp_cfs_rq->h_nr_queued -
				  grp_cfs_rq->h_nr_idle;
		if (!cfs_rq_is_idle(grp_cfs_rq))
			idle_task_delta *= -1;

		for_each_sched_entity(se) {
			struct cfs_rq *cfs_rq = cfs_rq_of(se);

			if (!se->on_rq)
				break;

			cfs_rq->h_nr_idle += idle_task_delta;

			/* Already accounted at parent level and above. */
			if (cfs_rq_is_idle(cfs_rq))
				break;
		}

next_cpu:
		rq_unlock_irqrestore(rq, &rf);
	}

	/* Idle groups have minimum weight. */
	if (tg_is_idle(tg))
		__sched_group_set_shares(tg, scale_load(WEIGHT_IDLEPRIO));
	else
		__sched_group_set_shares(tg, NICE_0_LOAD);

	mutex_unlock(&shares_mutex);
	return 0;
}

#endif /* CONFIG_FAIR_GROUP_SCHED */


/*
 * get_rr_interval_fair - 向通用时间片查询接口返回实体时间片的 jiffies 表示。
 * 公平类仍按 EEVDF 选取任务，此接口不表示采用实时轮转调度。
 */
static unsigned int get_rr_interval_fair(struct rq *rq, struct task_struct *task)
{
	struct sched_entity *se = &task->se;
	unsigned int rr_interval = 0;

	/*
	 * Time slice is 0 for SCHED_OTHER tasks that are on an otherwise
	 * idle runqueue:
	 */
	if (rq->cfs.load.weight)
		rr_interval = NS_TO_JIFFIES(se->slice);

	return rr_interval;
}

/*
 * All the scheduling class methods:
 */
/*
 * 调度器核心通过此回调表调用公平类：
 * 队列管理、抢占检查、选择、tick 和迁移
 * 各自负责一段状态转换。
 * 选择任务与真正的架构上下文切换由不同层完成。
 */
DEFINE_SCHED_CLASS(fair) = {

	/*
	 * 任务进入/离开队列，以及主动让出 CPU。
	 */
	.enqueue_task		= enqueue_task_fair,
	.dequeue_task		= dequeue_task_fair,
	.yield_task		= yield_task_fair,
	.yield_to_task		= yield_to_task_fair,

	.wakeup_preempt		= check_preempt_wakeup_fair,

	/*
	 * 选择候选与切换前后的实体记账。
	 */
	.pick_task		= pick_task_fair,
	.pick_next_task		= __pick_next_task_fair,
	.put_prev_task		= put_prev_task_fair,
	.set_next_task          = set_next_task_fair,

	/*
	 * 跨 CPU 放置和负载均衡。
	 */
	.balance		= balance_fair,
	.select_task_rq		= select_task_rq_fair,
	.migrate_task_rq	= migrate_task_rq_fair,

	.rq_online		= rq_online_fair,
	.rq_offline		= rq_offline_fair,

	.task_dead		= task_dead_fair,
	.set_cpus_allowed	= set_cpus_allowed_fair,

	/*
	 * 周期运行记账和任务生命周期事件。
	 */
	.task_tick		= task_tick_fair,
	.task_fork		= task_fork_fair,

	.reweight_task		= reweight_task_fair,
	.prio_changed		= prio_changed_fair,
	.switched_from		= switched_from_fair,
	.switched_to		= switched_to_fair,

	.get_rr_interval	= get_rr_interval_fair,

	.update_curr		= update_curr_fair,

#ifdef CONFIG_FAIR_GROUP_SCHED
	.task_change_group	= task_change_group_fair,
#endif

#ifdef CONFIG_SCHED_CORE
	.task_is_throttled	= task_is_throttled_fair,
#endif

#ifdef CONFIG_UCLAMP_TASK
	.uclamp_enabled		= 1,
#endif
};

/*
 * print_cfs_stats - 在 RCU 保护下输出指定 CPU 各叶子公平队列的统计。
 */
void print_cfs_stats(struct seq_file *m, int cpu)
{
	struct cfs_rq *cfs_rq, *pos;

	rcu_read_lock();
	for_each_leaf_cfs_rq_safe(cpu_rq(cpu), cfs_rq, pos)
		print_cfs_rq(m, cpu, cfs_rq);
	rcu_read_unlock();
}

#ifdef CONFIG_NUMA_BALANCING
/*
 * show_numa_stats - 按内存节点输出任务及其 NUMA 组的私有、共享访问统计。
 */
void show_numa_stats(struct task_struct *p, struct seq_file *m)
{
	int node;
	unsigned long tsf = 0, tpf = 0, gsf = 0, gpf = 0;
	struct numa_group *ng;

	rcu_read_lock();
	ng = rcu_dereference(p->numa_group);
	for_each_online_node(node) {
		if (p->numa_faults) {
			tsf = p->numa_faults[task_faults_idx(NUMA_MEM, node, 0)];
			tpf = p->numa_faults[task_faults_idx(NUMA_MEM, node, 1)];
		}
		if (ng) {
			gsf = ng->faults[task_faults_idx(NUMA_MEM, node, 0)],
			gpf = ng->faults[task_faults_idx(NUMA_MEM, node, 1)];
		}
		print_numa_stats(m, node, tsf, tpf, gsf, gpf);
	}
	rcu_read_unlock();
}
#endif /* CONFIG_NUMA_BALANCING */

/*
 * init_sched_fair_class - 分配每 CPU 均衡临时掩码，
 * 注册调度软中断并初始化带宽和 NO_HZ 辅助状态。
 */
__init void init_sched_fair_class(void)
{
	int i;

	for_each_possible_cpu(i) {
		zalloc_cpumask_var_node(&per_cpu(load_balance_mask, i), GFP_KERNEL, cpu_to_node(i));
		zalloc_cpumask_var_node(&per_cpu(select_rq_mask,    i), GFP_KERNEL, cpu_to_node(i));
		zalloc_cpumask_var_node(&per_cpu(should_we_balance_tmpmask, i),
					GFP_KERNEL, cpu_to_node(i));

#ifdef CONFIG_CFS_BANDWIDTH
		INIT_CSD(&cpu_rq(i)->cfsb_csd, __cfsb_csd_unthrottle, cpu_rq(i));
		INIT_LIST_HEAD(&cpu_rq(i)->cfsb_csd_list);
#endif
	}

	open_softirq(SCHED_SOFTIRQ, sched_balance_softirq);

#ifdef CONFIG_NO_HZ_COMMON
	nohz.next_balance = jiffies;
	nohz.next_blocked = jiffies;
	zalloc_cpumask_var(&nohz.idle_cpus_mask, GFP_NOWAIT);
#endif
}
