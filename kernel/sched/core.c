// SPDX-License-Identifier: GPL-2.0-only
/*
 *  kernel/sched/core.c
 *
 *  Core kernel CPU scheduler code
 *
 *  Copyright (C) 1991-2002  Linus Torvalds
 *  Copyright (C) 1998-2024  Ingo Molnar, Red Hat
 */

/*
 * 调度器核心阅读导航：
 *
 * 本文件负责各调度类共用的队列管理、唤醒、迁移和上下文切换机制；
 * 具体的任务排序与时间分配由 fair.c、rt.c、deadline.c、ext.c 等实现。
 *
 * 常见调用链：
 *   主动调度：schedule() -> __schedule_loop() -> __schedule()
 *   选择任务：__schedule() -> pick_next_task() -> 调度类选择回调
 *   切换执行：context_switch() -> switch_to() -> finish_task_switch()
 *   普通唤醒：try_to_wake_up() -> ttwu_queue() -> ttwu_do_activate()
 *             远端异步入队时，由目标 CPU 的 sched_ttwu_pending() 接续处理。
 *   首次运行：sched_fork() 等完成初始化，wake_up_new_task() 首次入队，
 *             新任务通过 schedule_tail() 完成首次切换的收尾。
 *
 * 唤醒使任务具备运行条件，并按需请求重调度，不保证立即执行；
 * 一次调度也可能继续选择原任务，此时不会发生寄存器和内核栈切换。
 * 阅读并发路径时，应结合下文的状态维度说明和 Serialization rules，
 * 特别区分任务请求睡眠、离开运行队列以及完全切出 CPU 这三个时刻。
 *
 * 按问题查找实现：
 *   1. 谁保护调度状态：Serialization rules、task_rq_lock()。
 *   2. 怎样计算执行时间：update_rq_clock()、sched_tick()。
 *   3. 怎样请求让出 CPU：resched_curr()、各个 schedule 入口。
 *   4. 怎样入队和出队：enqueue/dequeue_task()、activate/deactivate_task()。
 *   5. 怎样改变 CPU 归属：affine_move_task()、migration_cpu_stop()。
 *   6. 怎样唤醒睡眠者：try_to_wake_up() 及 ttwu_* 辅助函数。
 *   7. 怎样启动新任务：sched_fork()、wake_up_new_task()、schedule_tail()。
 *   8. 怎样选择和切换：pick_next_task()、__schedule()、context_switch()。
 *   9. 怎样管理组与带宽：sched_*_group()、tg_set_cfs_bandwidth()。
 *  10. 怎样回收地址空间并发编号：sched_mm_cid_*()。
 *
 * 几个贯穿全文的约定：
 *   - rq 属于 CPU，task_struct 属于任务；任务迁移会改变所属 rq。
 *   - TASK_RUNNING 的数值是 0，既包括正在执行，也包括等待 CPU；
 *     它不等价于 rq->curr == p，也不表示任务一定已经入队。
 *   - ENQUEUE_* / DEQUEUE_* 描述本次队列操作的原因与记账方式；
 *     WF_* 描述唤醒来源和放置条件，不能把两组标志混用。
 *   - “禁止抢占”稳住本 CPU，“关闭中断”防止本 CPU 中断重入，
 *     自旋锁负责跨 CPU 互斥；三者各自解决不同的并发问题。
 *   - 调度类负责策略，本文件把类回调与时钟、PSI、uclamp、RCU、
 *     地址空间和体系结构切换接起来。阅读单个回调时需同时看 sched.h。
 *
 * 以下注释解释各 CONFIG 分支的语义；某条路径是否参与当前构建，
 * 应以本仓库 .config 为准，不能仅凭函数出现在文件中判断。
 *
 * 初读建议沿两条主线往返看，而不必一次读完全部配置分支：
 *   “睡下去”：schedule() -> __schedule() -> try_to_block_task()
 *               -> pick_next_task() -> context_switch()。
 *   “叫起来”：wake_up_process() -> try_to_wake_up() -> ttwu_queue()
 *               -> ttwu_do_activate() -> wakeup_preempt()。
 * 两条线在同一个任务上并发相遇，下面的锁和内存屏障就是交接协议。
 * 再看迁移、核心调度和组控制时，可以追问：它改变了哪个任务状态、
 * 哪个 CPU 的队列，以及哪一方负责把未完成的操作接着做完？
 *
 * 常见缩写与语法：
 *   rq：每个逻辑 CPU 的运行队列；SMT 兄弟是共享物理核的逻辑 CPU。
 *   IPI：CPU 间中断，通知远端处理请求；LLC：最后一级缓存。
 *   PELT：对任务/队列负载及利用率做随时间衰减的统计。
 *   PSI：统计 CPU、内存和 I/O 等资源压力造成的停顿。
 *   RCU：使读者可并发访问对象，回收方等旧读者退出后再释放；
 *        它不自动保证被读对象的所有字段都不变化。
 *   guard()/scoped_guard()：作用域管理，离开作用域时自动解锁或
 *        恢复相应状态；没看见显式 unlock 不代表遗漏了解锁。
 */
#define INSTANTIATE_EXPORTED_MIGRATE_DISABLE
#include <linux/sched.h>
#include <linux/highmem.h>
#include <linux/hrtimer_api.h>
#include <linux/ktime_api.h>
#include <linux/sched/signal.h>
#include <linux/syscalls_api.h>
#include <linux/debug_locks.h>
#include <linux/prefetch.h>
#include <linux/capability.h>
#include <linux/pgtable_api.h>
#include <linux/wait_bit.h>
#include <linux/jiffies.h>
#include <linux/spinlock_api.h>
#include <linux/cpumask_api.h>
#include <linux/lockdep_api.h>
#include <linux/hardirq.h>
#include <linux/softirq.h>
#include <linux/refcount_api.h>
#include <linux/topology.h>
#include <linux/sched/clock.h>
#include <linux/sched/cond_resched.h>
#include <linux/sched/cputime.h>
#include <linux/sched/debug.h>
#include <linux/sched/hotplug.h>
#include <linux/sched/init.h>
#include <linux/sched/isolation.h>
#include <linux/sched/loadavg.h>
#include <linux/sched/mm.h>
#include <linux/sched/nohz.h>
#include <linux/sched/rseq_api.h>
#include <linux/sched/rt.h>

#include <linux/blkdev.h>
#include <linux/context_tracking.h>
#include <linux/cpuset.h>
#include <linux/delayacct.h>
#include <linux/init_task.h>
#include <linux/interrupt.h>
#include <linux/ioprio.h>
#include <linux/kallsyms.h>
#include <linux/kcov.h>
#include <linux/kprobes.h>
#include <linux/llist_api.h>
#include <linux/mmu_context.h>
#include <linux/mmzone.h>
#include <linux/mutex_api.h>
#include <linux/nmi.h>
#include <linux/nospec.h>
#include <linux/perf_event_api.h>
#include <linux/profile.h>
#include <linux/psi.h>
#include <linux/rcuwait_api.h>
#include <linux/rseq.h>
#include <linux/sched/wake_q.h>
#include <linux/scs.h>
#include <linux/slab.h>
#include <linux/syscalls.h>
#include <linux/vtime.h>
#include <linux/wait_api.h>
#include <linux/workqueue_api.h>
#include <linux/livepatch_sched.h>

#ifdef CONFIG_PREEMPT_DYNAMIC
# ifdef CONFIG_GENERIC_IRQ_ENTRY
#  include <linux/irq-entry-common.h>
# endif
#endif

#include <uapi/linux/sched/types.h>

#include <asm/irq_regs.h>
#include <asm/switch_to.h>
#include <asm/tlb.h>

#define CREATE_TRACE_POINTS
#include <linux/sched/rseq_api.h>
#include <trace/events/sched.h>
#include <trace/events/ipi.h>
#undef CREATE_TRACE_POINTS

#include "sched.h"
#include "stats.h"

#include "autogroup.h"
#include "pelt.h"
#include "smp.h"

#include "../workqueue_internal.h"
#include "../../io_uring/io-wq.h"
#include "../smpboot.h"
#include "../locking/mutex.h"

EXPORT_TRACEPOINT_SYMBOL_GPL(ipi_send_cpu);
EXPORT_TRACEPOINT_SYMBOL_GPL(ipi_send_cpumask);

/*
 * Export tracepoints that act as a bare tracehook (ie: have no trace event
 * associated with them) to allow external modules to probe them.
 */
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_cfs_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_rt_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_dl_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_irq_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_se_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(pelt_hw_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_cpu_capacity_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_overutilized_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_util_est_cfs_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_util_est_se_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_update_nr_running_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_compute_energy_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_entry_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_exit_tp);
EXPORT_TRACEPOINT_SYMBOL_GPL(sched_set_need_resched_tp);

/*
 * 每个 CPU 都有自己的 rq，内含各调度类的子队列及公共统计，而非一条
 * 所有任务共用的链表。rq->curr 记录执行者，rq->donor 记录调度资源提供者；
 * 未发生代理执行时二者通常相同。
 */
DEFINE_PER_CPU_SHARED_ALIGNED(struct rq, runqueues);
DEFINE_PER_CPU(struct rnd_state, sched_rnd_state);

#ifdef CONFIG_SCHED_PROXY_EXEC
DEFINE_STATIC_KEY_TRUE(__sched_proxy_exec);
/*
 * setup_proxy_exec - 解析启动参数，启用或禁用代理执行调度。
 * 返回 1 表示参数已处理，解析失败时返回 0。
 */
static int __init setup_proxy_exec(char *str)
{
	bool proxy_enable = true;

	if (*str && kstrtobool(str + 1, &proxy_enable)) {
		pr_warn("Unable to parse sched_proxy_exec=\n");
		return 0;
	}

	if (proxy_enable) {
		pr_info("sched_proxy_exec enabled via boot arg\n");
		static_branch_enable(&__sched_proxy_exec);
	} else {
		pr_info("sched_proxy_exec disabled via boot arg\n");
		static_branch_disable(&__sched_proxy_exec);
	}
	return 1;
}
#else
/*
 * setup_proxy_exec - 未编译代理执行支持时报告启动参数不可用。
 */
static int __init setup_proxy_exec(char *str)
{
	pr_warn("CONFIG_SCHED_PROXY_EXEC=n, so it cannot be enabled or disabled at boot time\n");
	return 0;
}
#endif
__setup("sched_proxy_exec", setup_proxy_exec);

/*
 * Debugging: various feature bits
 *
 * If SCHED_DEBUG is disabled, each compilation unit has its own copy of
 * sysctl_sched_features, defined in sched.h, to allow constants propagation
 * at compile time and compiler optimization based on features default.
 */
#define SCHED_FEAT(name, enabled)	\
	(1UL << __SCHED_FEAT_##name) * enabled |
__read_mostly unsigned int sysctl_sched_features =
#include "features.h"
	0;
#undef SCHED_FEAT

/*
 * Print a warning if need_resched is set for the given duration (if
 * LATENCY_WARN is enabled).
 *
 * If sysctl_resched_latency_warn_once is set, only one warning will be shown
 * per boot.
 */
__read_mostly int sysctl_resched_latency_warn_ms = 100;
__read_mostly int sysctl_resched_latency_warn_once = 1;

/*
 * Number of tasks to iterate in a single balance run.
 * Limited because this is done with IRQs disabled.
 */
__read_mostly unsigned int sysctl_sched_nr_migrate = SCHED_NR_MIGRATE_BREAK;

__read_mostly int scheduler_running;

#ifdef CONFIG_SCHED_CORE

DEFINE_STATIC_KEY_FALSE(__sched_core_enabled);

/* kernel prio, less is more */
/*
 * __task_prio - 将任务调度类映射为核心调度使用的优先级，
 * 数值越小越优先。
 */
static inline int __task_prio(const struct task_struct *p)
{
	if (p->sched_class == &stop_sched_class) /* trumps deadline */
		return -2;

	if (p->dl_server)
		return -1; /* deadline */

	if (rt_or_dl_prio(p->prio))
		return p->prio; /* [-1, 99] */

	if (p->sched_class == &idle_sched_class)
		return MAX_RT_PRIO + NICE_WIDTH; /* 140 */

	if (task_on_scx(p))
		return MAX_RT_PRIO + MAX_NICE + 1; /* 120, squash ext */

	return MAX_RT_PRIO + MAX_NICE; /* 119, squash fair */
}

/*
 * l(a,b)
 * le(a,b) := !l(b,a)
 * g(a,b)  := l(b,a)
 * ge(a,b) := !l(a,b)
 */

/* real prio, less is less */
/*
 * prio_less - 比较任务的实际调度优先级，a 低于 b 时返回 true。
 * 同类任务按截止时间或对应调度类的规则比较，并考虑强制空闲状态。
 */
static inline bool prio_less(const struct task_struct *a,
			     const struct task_struct *b, bool in_fi)
{

	int pa = __task_prio(a), pb = __task_prio(b);

	if (-pa < -pb)
		return true;

	if (-pb < -pa)
		return false;

	if (pa == -1) { /* dl_prio() doesn't work because of stop_class above */
		const struct sched_dl_entity *a_dl, *b_dl;

		a_dl = &a->dl;
		/*
		 * Since,'a' and 'b' can be CFS tasks served by DL server,
		 * __task_prio() can return -1 (for DL) even for those. In that
		 * case, get to the dl_server's DL entity.
		 */
		if (a->dl_server)
			a_dl = a->dl_server;

		b_dl = &b->dl;
		if (b->dl_server)
			b_dl = b->dl_server;

		return !dl_time_before(a_dl->deadline, b_dl->deadline);
	}

	if (pa == MAX_RT_PRIO + MAX_NICE)	/* fair */
		return cfs_prio_less(a, b, in_fi);

#ifdef CONFIG_SCHED_CLASS_EXT
	if (pa == MAX_RT_PRIO + MAX_NICE + 1)	/* ext */
		return scx_prio_less(a, b, in_fi);
#endif

	return false;
}

/*
 * __sched_core_less - 按核心 cookie 和任务优先级确定核心调度树中的顺序。
 */
/*
 * 核心树是每个 rq 上的辅助索引，不替代调度类自己的队列。
 * 第一关键字 cookie 将兼容任务放在连续区间，第二关键字把高优先级
 * 任务排在区间左端，因而可先定位 cookie，再跳过被带宽限流的任务。
 * cookie 为 0 的任务不进该树；普通候选仍由调度类的 pick_task 取得。
 */
static inline bool __sched_core_less(const struct task_struct *a,
				     const struct task_struct *b)
{
	if (a->core_cookie < b->core_cookie)
		return true;

	if (a->core_cookie > b->core_cookie)
		return false;

	/* flip prio, so high prio is leftmost */
	if (prio_less(b, a, !!task_rq(a)->core->core_forceidle_count))
		return true;

	return false;
}

#define __node_2_sc(node) rb_entry((node), struct task_struct, core_node)

/*
 * rb_sched_core_less - 将红黑树节点转换为任务，执行核心调度排序比较。
 */
static inline bool rb_sched_core_less(struct rb_node *a, const struct rb_node *b)
{
	return __sched_core_less(__node_2_sc(a), __node_2_sc(b));
}

/*
 * rb_sched_core_cmp - 比较目标 cookie 与树中任务的 cookie，供红黑树查找使用。
 */
static inline int rb_sched_core_cmp(const void *key, const struct rb_node *node)
{
	const struct task_struct *p = __node_2_sc(node);
	unsigned long cookie = (unsigned long)key;

	if (cookie < p->core_cookie)
		return -1;

	if (cookie > p->core_cookie)
		return 1;

	return 0;
}

/*
 * sched_core_enqueue - 更新核心任务序列，
 * 并将带 cookie 的非延迟任务加入核心树。
 * 调用者须持有运行队列锁。
 */
void sched_core_enqueue(struct rq *rq, struct task_struct *p)
{
	if (p->se.sched_delayed)
		return;

	/* 使此前缓存的整核选择失效，包括不进入 cookie 树的普通任务。 */
	rq->core->core_task_seq++;

	if (!p->core_cookie)
		return;

	rb_add(&p->core_node, &rq->core_tree, rb_sched_core_less);
}

/*
 * sched_core_dequeue - 从核心树移除任务，
 * 并在必要时触发强制空闲状态的重新核算。
 * 调用者须持有运行队列锁。
 */
void sched_core_dequeue(struct rq *rq, struct task_struct *p, int flags)
{
	if (p->se.sched_delayed)
		return;

	rq->core->core_task_seq++;

	if (sched_core_enqueued(p)) {
		rb_erase(&p->core_node, &rq->core_tree);
		RB_CLEAR_NODE(&p->core_node);
	}

	/*
	 * Migrating the last task off the cpu, with the cpu in forced idle
	 * state. Reschedule to create an accounting edge for forced idle,
	 * and re-examine whether the core is still in forced idle state.
	 */
	if (!(flags & DEQUEUE_SAVE) && rq->nr_running == 1 &&
	    rq->core->core_forceidle_count && rq->curr == rq->idle)
		resched_curr(rq);
}

/*
 * sched_task_is_throttled - 查询调度类，判断任务在指定 CPU 上是否受到限流。
 */
static int sched_task_is_throttled(struct task_struct *p, int cpu)
{
	if (p->sched_class->task_is_throttled)
		return p->sched_class->task_is_throttled(p, cpu);

	return 0;
}

/*
 * sched_core_next - 查找同一 cookie 的下一个未限流任务，无匹配项时返回 NULL。
 */
static struct task_struct *sched_core_next(struct task_struct *p, unsigned long cookie)
{
	struct rb_node *node = &p->core_node;
	int cpu = task_cpu(p);

	do {
		node = rb_next(node);
		if (!node)
			return NULL;

		p = __node_2_sc(node);
		if (p->core_cookie != cookie)
			return NULL;

	} while (sched_task_is_throttled(p, cpu));

	return p;
}

/*
 * Find left-most (aka, highest priority) and unthrottled task matching @cookie.
 * If no suitable task is found, NULL will be returned.
 */
/*
 * sched_core_find - 返回核心树中匹配 cookie 的最高优先级未限流任务。
 */
static struct task_struct *sched_core_find(struct rq *rq, unsigned long cookie)
{
	struct task_struct *p;
	struct rb_node *node;

	node = rb_find_first((void *)cookie, &rq->core_tree, rb_sched_core_cmp);
	if (!node)
		return NULL;

	p = __node_2_sc(node);
	if (!sched_task_is_throttled(p, rq->cpu))
		return p;

	return sched_core_next(p, cookie);
}

/*
 * Magic required such that:
 *
 *	raw_spin_rq_lock(rq);
 *	...
 *	raw_spin_rq_unlock(rq);
 *
 * ends up locking and unlocking the _same_ lock, and all CPUs
 * always agree on what rq has what lock.
 *
 * XXX entirely possible to selectively enable cores, don't bother for now.
 */

static DEFINE_MUTEX(sched_core_mutex);
static atomic_t sched_core_count;
static struct cpumask sched_core_mask;

/*
 * sched_core_lock - 保存中断状态并依次锁定同一物理核心
 * 的所有 SMT 运行队列。
 */
static void sched_core_lock(int cpu, unsigned long *flags)
{
	const struct cpumask *smt_mask = cpu_smt_mask(cpu);
	int t, i = 0;

	local_irq_save(*flags);
	for_each_cpu(t, smt_mask)
		raw_spin_lock_nested(&cpu_rq(t)->__lock, i++);
}

/*
 * sched_core_unlock - 释放同一物理核心的全部运行队列锁，并恢复中断状态。
 */
static void sched_core_unlock(int cpu, unsigned long *flags)
{
	const struct cpumask *smt_mask = cpu_smt_mask(cpu);
	int t;

	for_each_cpu(t, smt_mask)
		raw_spin_unlock(&cpu_rq(t)->__lock);
	local_irq_restore(*flags);
}

/*
 * __sched_core_flip - 在 CPU 热插拔保护下逐核心切换核心调度状态。
 * 等待正在进行的核心级任务选择结束，避免运行队列锁绑定发生竞态。
 */
static void __sched_core_flip(bool enabled)
{
	unsigned long flags;
	int cpu, t;

	cpus_read_lock();

	/*
	 * Toggle the online cores, one by one.
	 */
	cpumask_copy(&sched_core_mask, cpu_online_mask);
	for_each_cpu(cpu, &sched_core_mask) {
		const struct cpumask *smt_mask = cpu_smt_mask(cpu);

		sched_core_lock(cpu, &flags);

		/*
		 * A core-wide selection may have the shared rq lock temporarily
		 * released by a lock-dropping ->pick_task(). Flipping would
		 * rebind rq_lockp() under it. Wait it out.
		 */
		while (cpu_rq(cpu)->core->core_pick_in_flight) {
			sched_core_unlock(cpu, &flags);
			cpu_relax();
			sched_core_lock(cpu, &flags);
		}

		for_each_cpu(t, smt_mask)
			cpu_rq(t)->core_enabled = enabled;

		cpu_rq(cpu)->core->core_forceidle_start = 0;

		sched_core_unlock(cpu, &flags);

		cpumask_andnot(&sched_core_mask, &sched_core_mask, smt_mask);
	}

	/*
	 * Toggle the offline CPUs.
	 */
	for_each_cpu_andnot(cpu, cpu_possible_mask, cpu_online_mask)
		cpu_rq(cpu)->core_enabled = enabled;

	cpus_read_unlock();
}

/*
 * sched_core_assert_empty - 检查所有 CPU 的核心调度红黑树是否为空，
 * 异常时告警。
 */
static void sched_core_assert_empty(void)
{
	int cpu;

	for_each_possible_cpu(cpu)
		WARN_ON_ONCE(!RB_EMPTY_ROOT(&cpu_rq(cpu)->core_tree));
}

/*
 * __sched_core_enable - 启用核心调度静态分支，
 * 经 RCU 同步后切换运行队列锁模式。
 */
static void __sched_core_enable(void)
{
	/*
	 * 先让取锁者进入能重试锁指针的慢路径，等待旧取锁临界区退出，
	 * 最后重绑为共享锁。顺序反过来会让同一 rq 被两把锁分别保护。
	 */
	static_branch_enable(&__sched_core_enabled);
	/*
	 * Ensure all previous instances of raw_spin_rq_*lock() have finished
	 * and future ones will observe !sched_core_disabled().
	 */
	synchronize_rcu();
	__sched_core_flip(true);
	sched_core_assert_empty();
}

/*
 * __sched_core_disable - 确认核心树为空，
 * 恢复独立运行队列锁并关闭核心调度分支。
 */
static void __sched_core_disable(void)
{
	sched_core_assert_empty();
	__sched_core_flip(false);
	static_branch_disable(&__sched_core_enabled);
}

/*
 * sched_core_get - 增加核心调度使用引用，首个使用者负责启用核心调度。
 */
void sched_core_get(void)
{
	if (atomic_inc_not_zero(&sched_core_count))
		return;

	mutex_lock(&sched_core_mutex);
	if (!atomic_read(&sched_core_count))
		__sched_core_enable();

	smp_mb__before_atomic();
	atomic_inc(&sched_core_count);
	mutex_unlock(&sched_core_mutex);
}

/*
 * __sched_core_put - 在工作队列上下文释放引用，
 * 最后一个引用消失时关闭核心调度。
 */
static void __sched_core_put(struct work_struct *work)
{
	if (atomic_dec_and_mutex_lock(&sched_core_count, &sched_core_mutex)) {
		__sched_core_disable();
		mutex_unlock(&sched_core_mutex);
	}
}

/*
 * sched_core_put - 释放核心调度引用，
 * 将可能关闭功能的最后一次释放推迟到工作队列。
 */
void sched_core_put(void)
{
	static DECLARE_WORK(_work, __sched_core_put);

	/*
	 * "There can be only one"
	 *
	 * Either this is the last one, or we don't actually need to do any
	 * 'work'. If it is the last *again*, we rely on
	 * WORK_STRUCT_PENDING_BIT.
	 */
	if (!atomic_add_unless(&sched_core_count, -1, 1))
		schedule_work(&_work);
}

#else /* !CONFIG_SCHED_CORE: */

/*
 * sched_core_enqueue - 未启用核心调度时的空操作入队接口。
 */
static inline void sched_core_enqueue(struct rq *rq, struct task_struct *p) { }
/*
 * sched_core_dequeue - 未启用核心调度时的空操作出队接口。
 */
static inline void
sched_core_dequeue(struct rq *rq, struct task_struct *p, int flags) { }

#endif /* !CONFIG_SCHED_CORE */

/* need a wrapper since we may need to trace from modules */
EXPORT_TRACEPOINT_SYMBOL(sched_set_state_tp);

/* Call via the helper macro trace_set_current_state. */
/*
 * __trace_set_current_state - 为当前任务发出状态变更跟踪事件，
 * 供辅助宏和模块使用。
 */
void __trace_set_current_state(int state_value)
{
	trace_sched_set_state_tp(current, state_value);
}
EXPORT_SYMBOL(__trace_set_current_state);

/*
 * 阅读调度与唤醒路径时，需区分任务状态的三个不同维度：
 *   __state：任务是否请求睡眠，以及哪些事件可以唤醒它；
 *   on_rq：是否在运行队列上，或处于跨队列迁移的中间阶段；
 *   on_cpu：是否被 CPU 认领，涵盖切入准备、执行及切出收尾。
 * 因此，设置睡眠状态不等于已经出队，出队也不等于已经切出 CPU。
 * pi_lock 串行化同一任务的唤醒和属性修改，rq 锁保护队列及调度选择；
 * 不能用其中一个状态位代替另一个状态位或相应的锁。
 *
 * 普通任务的几种典型快照（不含代理执行和延迟出队）：
 *   __state          on_rq       on_cpu   含义
 *   TASK_RUNNING     QUEUED      0        已就绪，等待 CPU
 *   TASK_RUNNING     QUEUED      1        正在执行或处于切换交接窗口
 *   睡眠状态         QUEUED      1        已请求睡眠，尚未由调度器出队
 *   睡眠状态         0           1        已出队，旧 CPU 仍在完成切出
 *   睡眠状态         0           0        已完全阻塞
 *   TASK_WAKING      0           0 或 1   唤醒已认领，等待切出/入队
 * 这只是解释状态含义，不是无锁检查规则：分别读取三个字段可能得到
 * 来自不同时刻的值。需要可靠结论时必须使用相应锁或配对的内存序。
 */

/*
 * Serialization rules:
 *
 * Lock order:
 *
 *   p->pi_lock
 *     rq->lock
 *       hrtimer_cpu_base->lock (hrtimer_start() for bandwidth controls)
 *
 *  rq1->lock
 *    rq2->lock  where: rq1 < rq2
 *
 * Regular state:
 *
 * Normal scheduling state is serialized by rq->lock. __schedule() takes the
 * local CPU's rq->lock, it optionally removes the task from the runqueue and
 * always looks at the local rq data structures to find the most eligible task
 * to run next.
 *
 * Task enqueue is also under rq->lock, possibly taken from another CPU.
 * Wakeups from another LLC domain might use an IPI to transfer the enqueue to
 * the local CPU to avoid bouncing the runqueue state around [ see
 * ttwu_queue_wakelist() ]
 *
 * Task wakeup, specifically wakeups that involve migration, are horribly
 * complicated to avoid having to take two rq->locks.
 *
 * Special state:
 *
 * System-calls and anything external will use task_rq_lock() which acquires
 * both p->pi_lock and rq->lock. As a consequence the state they change is
 * stable while holding either lock:
 *
 *  - sched_setaffinity()/
 *    set_cpus_allowed_ptr():	p->cpus_ptr, p->nr_cpus_allowed
 *  - set_user_nice():		p->se.load, p->*prio
 *  - __sched_setscheduler():	p->sched_class, p->policy, p->*prio,
 *				p->se.load, p->rt_priority,
 *				p->dl.dl_{runtime, deadline, period, flags, bw, density}
 *  - sched_setnuma():		p->numa_preferred_nid
 *  - sched_move_task():	p->sched_task_group
 *  - uclamp_update_active()	p->uclamp*
 *
 * p->state <- TASK_*:
 *
 *   is changed locklessly using set_current_state(), __set_current_state() or
 *   set_special_state(), see their respective comments, or by
 *   try_to_wake_up(). This latter uses p->pi_lock to serialize against
 *   concurrent self.
 *
 * p->on_rq <- { 0, 1 = TASK_ON_RQ_QUEUED, 2 = TASK_ON_RQ_MIGRATING }:
 *
 *   is set by activate_task() and cleared by deactivate_task(), under
 *   rq->lock. Non-zero indicates the task is runnable, the special
 *   ON_RQ_MIGRATING state is used for migration without holding both
 *   rq->locks. It indicates task_cpu() is not stable, see task_rq_lock().
 *
 *   Additionally it is possible to be ->on_rq but still be considered not
 *   runnable when p->se.sched_delayed is true. These tasks are on the runqueue
 *   but will be dequeued as soon as they get picked again. See the
 *   task_is_runnable() helper.
 *
 * p->on_cpu <- { 0, 1 }:
 *
 *   is set by prepare_task() and cleared by finish_task() such that it will be
 *   set before p is scheduled-in and cleared after p is scheduled-out, both
 *   under rq->lock. Non-zero indicates the task is running on its CPU.
 *
 *   [ The astute reader will observe that it is possible for two tasks on one
 *     CPU to have ->on_cpu = 1 at the same time. ]
 *
 * task_cpu(p): is changed by set_task_cpu(), the rules are:
 *
 *  - Don't call set_task_cpu() on a blocked task:
 *
 *    We don't care what CPU we're not running on, this simplifies hotplug,
 *    the CPU assignment of blocked tasks isn't required to be valid.
 *
 *  - for try_to_wake_up(), called under p->pi_lock:
 *
 *    This allows try_to_wake_up() to only take one rq->lock, see its comment.
 *
 *  - for migration called under rq->lock:
 *    [ see task_on_rq_migrating() in task_rq_lock() ]
 *
 *    o move_queued_task()
 *    o detach_task()
 *
 *  - for migration called under double_rq_lock():
 *
 *    o __migrate_swap_task()
 *    o push_rt_task() / pull_rt_task()
 *    o push_dl_task() / pull_dl_task()
 *    o dl_task_offline_migration()
 *
 */

/*
 * raw_spin_rq_lock_nested - 按锁嵌套层级获取运行队列锁，
 * 兼容核心调度锁的动态切换。
 */
void raw_spin_rq_lock_nested(struct rq *rq, int subclass)
{
	raw_spinlock_t *lock;

	/* Matches synchronize_rcu() in __sched_core_enable() */
	preempt_disable();
	if (sched_core_disabled()) {
		raw_spin_lock_nested(&rq->__lock, subclass);
		/* preempt_count *MUST* be > 1 */
		preempt_enable_no_resched();
		return;
	}

	for (;;) {
		lock = __rq_lockp(rq);
		raw_spin_lock_nested(lock, subclass);
		/*
		 * 等锁期间可能切换了核心调度，rq 也就改用了另一把锁。
		 * 拿到旧锁并不算成功，必须确认它仍是当前 rq 对应的锁。
		 */
		if (likely(lock == __rq_lockp(rq))) {
			/* preempt_count *MUST* be > 1 */
			preempt_enable_no_resched();
			return;
		}
		raw_spin_unlock(lock);
	}
}

/*
 * raw_spin_rq_trylock - 尝试获取运行队列当前对应的锁，成功时返回 true。
 */
bool raw_spin_rq_trylock(struct rq *rq)
{
	raw_spinlock_t *lock;
	bool ret;

	/* Matches synchronize_rcu() in __sched_core_enable() */
	preempt_disable();
	if (sched_core_disabled()) {
		ret = raw_spin_trylock(&rq->__lock);
		preempt_enable();
		return ret;
	}

	for (;;) {
		lock = __rq_lockp(rq);
		ret = raw_spin_trylock(lock);
		if (!ret || (likely(lock == __rq_lockp(rq)))) {
			preempt_enable();
			return ret;
		}
		raw_spin_unlock(lock);
	}
}

/*
 * raw_spin_rq_unlock - 释放运行队列实际使用的自旋锁，包括核心共享锁。
 */
void raw_spin_rq_unlock(struct rq *rq)
{
	raw_spin_unlock(rq_lockp(rq));
}

/*
 * double_rq_lock - safely lock two runqueues
 */
/*
 * double_rq_lock - 按统一顺序锁定两个运行队列，
 * 避免重复获取同一核心共享锁。
 * 调用前必须关闭中断。
 */
void double_rq_lock(struct rq *rq1, struct rq *rq2)
{
	lockdep_assert_irqs_disabled();

	/*
	 * 两个 CPU 同时互相搬任务时，也必须按同一顺序拿锁，否则会
	 * 形成 A 等 B、B 等 A。核心调度下两个 rq 还可能共用一把锁，
	 * 所以下面比较实际锁指针，不能仅凭 rq1 != rq2 就再锁一次。
	 */
	if (rq_order_less(rq2, rq1))
		swap(rq1, rq2);

	raw_spin_rq_lock(rq1);
	if (__rq_lockp(rq1) != __rq_lockp(rq2))
		raw_spin_rq_lock_nested(rq2, SINGLE_DEPTH_NESTING);

	double_rq_clock_clear_update(rq1, rq2);
}

/*
 * __task_rq_lock - lock the rq @p resides on.
 */
/*
 * __task_rq_lock - 锁定任务当前所属的运行队列，并重试以避开任务迁移。
 * 调用者须已持有任务的 pi_lock；返回的运行队列保持加锁状态。
 */
struct rq *__task_rq_lock(struct task_struct *p, struct rq_flags *rf)
	__acquires(rq->lock)
{
	struct rq *rq;

	lockdep_assert_held(&p->pi_lock);

	for (;;) {
		rq = task_rq(p);
		raw_spin_rq_lock(rq);
		/*
		 * 读 task_rq() 到拿锁之间任务可能迁走；即便 CPU 编号恰好
		 * 相同，也必须排除 MIGRATING，因为目标端可能尚未完成入队。
		 */
		if (likely(rq == task_rq(p) && !task_on_rq_migrating(p))) {
			rq_pin_lock(rq, rf);
			return rq;
		}
		raw_spin_rq_unlock(rq);

		while (unlikely(task_on_rq_migrating(p)))
			cpu_relax();
	}
}

/*
 * task_rq_lock - lock p->pi_lock and lock the rq @p resides on.
 */
/*
 * task_rq_lock - 保存中断状态，获取任务 pi_lock 及其稳定所属的运行队列锁。
 */
struct rq *task_rq_lock(struct task_struct *p, struct rq_flags *rf)
	__acquires(p->pi_lock)
	__acquires(rq->lock)
{
	struct rq *rq;

	for (;;) {
		/*
		 * pi_lock 在外、rq 锁在内。pi_lock 稳住唤醒和属性修改，
		 * 但迁移者可能只持 rq 锁，故下面仍要验证归属并按需重试。
		 * rf 保存 IRQ 状态和 lockdep 的锁固定信息，解锁时配对恢复。
		 */
		raw_spin_lock_irqsave(&p->pi_lock, rf->flags);
		rq = task_rq(p);
		raw_spin_rq_lock(rq);
		/*
		 *	move_queued_task()		task_rq_lock()
		 *
		 *	ACQUIRE (rq->lock)
		 *	[S] ->on_rq = MIGRATING		[L] rq = task_rq()
		 *	WMB (__set_task_cpu())		ACQUIRE (rq->lock);
		 *	[S] ->cpu = new_cpu		[L] task_rq()
		 *					[L] ->on_rq
		 *	RELEASE (rq->lock)
		 *
		 * If we observe the old CPU in task_rq_lock(), the acquire of
		 * the old rq->lock will fully serialize against the stores.
		 *
		 * If we observe the new CPU in task_rq_lock(), the address
		 * dependency headed by '[L] rq = task_rq()' and the acquire
		 * will pair with the WMB to ensure we then also see migrating.
		 */
		if (likely(rq == task_rq(p) && !task_on_rq_migrating(p))) {
			rq_pin_lock(rq, rf);
			return rq;
		}
		raw_spin_rq_unlock(rq);
		raw_spin_unlock_irqrestore(&p->pi_lock, rf->flags);

		while (unlikely(task_on_rq_migrating(p)))
			cpu_relax();
	}
}

/*
 * RQ-clock updating methods:
 */

/*
 * update_rq_clock_task - 从时间增量中扣除中断和虚拟机偷取时间，
 * 更新任务及 PELT 时钟。
 */
static void update_rq_clock_task(struct rq *rq, s64 delta)
{
/*
 * In theory, the compile should just see 0 here, and optimize out the call
 * to sched_rt_avg_update. But I don't trust it...
 */
	s64 __maybe_unused steal = 0, irq_delta = 0;

	/*
	 * delta 是本 rq 距离上次更新经过的时间；只有扣掉可单独统计的
	 * IRQ/steal 部分后，剩余量才加入 clock_task。它仍是队列时钟，
	 * 并非某个任务的累计运行时间；后者由调度类 update_curr() 结算。
	 */
#ifdef CONFIG_IRQ_TIME_ACCOUNTING
	if (irqtime_enabled()) {
		irq_delta = irq_time_read(cpu_of(rq)) - rq->prev_irq_time;

		/*
		 * Since irq_time is only updated on {soft,}irq_exit, we might run into
		 * this case when a previous update_rq_clock() happened inside a
		 * {soft,}IRQ region.
		 *
		 * When this happens, we stop ->clock_task and only update the
		 * prev_irq_time stamp to account for the part that fit, so that a next
		 * update will consume the rest. This ensures ->clock_task is
		 * monotonic.
		 *
		 * It does however cause some slight miss-attribution of {soft,}IRQ
		 * time, a more accurate solution would be to update the irq_time using
		 * the current rq->clock timestamp, except that would require using
		 * atomic ops.
		 *
		 * 中断耗时可能在退出中断时才一次性补记，超过本轮 delta。
		 * 本轮最多扣完 delta；只推进已扣部分的 prev_irq_time，
		 * 把余量留给后续更新，避免任务时钟倒退或重复扣账。
		 */
		if (irq_delta > delta)
			irq_delta = delta;

		rq->prev_irq_time += irq_delta;
		delta -= irq_delta;
		delayacct_irq(rq->curr, irq_delta);
	}
#endif
#ifdef CONFIG_PARAVIRT_TIME_ACCOUNTING
	if (static_key_false((&paravirt_steal_rq_enabled))) {
		u64 prev_steal;

		steal = prev_steal = paravirt_steal_clock(cpu_of(rq));
		steal -= rq->prev_steal_time_rq;

		if (unlikely(steal > delta))
			steal = delta;

		rq->prev_steal_time_rq = prev_steal;
		delta -= steal;
	}
#endif

	rq->clock_task += delta;

#ifdef CONFIG_HAVE_SCHED_AVG_IRQ
	if ((irq_delta + steal) && sched_feat(NONTASK_CAPACITY))
		update_irq_load_avg(rq, irq_delta + steal);
#endif
	update_rq_clock_pelt(rq, delta);
}

/*
 * update_rq_clock - 在运行队列锁保护下更新队列时钟，
 * 并保持任务时钟单调递增。
 */
void update_rq_clock(struct rq *rq)
{
	s64 delta;
	u64 clock;

	lockdep_assert_rq_held(rq);

	if (rq->clock_update_flags & RQCF_ACT_SKIP)
		return;

	if (sched_feat(WARN_DOUBLE_CLOCK))
		WARN_ON_ONCE(rq->clock_update_flags & RQCF_UPDATED);
	rq->clock_update_flags |= RQCF_UPDATED;

	clock = sched_clock_cpu(cpu_of(rq));
	scx_rq_clock_update(rq, clock);

	/*
	 * 不接受倒退的时钟样本。clock 记录队列时间，clock_task 再扣除
	 * 可单独统计的中断及虚拟机偷取时间，避免把这些时间算给任务。
	 */
	delta = clock - rq->clock;
	if (delta < 0)
		return;
	rq->clock += delta;

	update_rq_clock_task(rq, delta);
}

#ifdef CONFIG_SCHED_HRTICK
/*
 * Use HR-timers to deliver accurate preemption points.
 */

/*
 * hrtick_clear - 取消运行队列上仍处于活动状态的高精度调度定时器。
 */
static void hrtick_clear(struct rq *rq)
{
	if (hrtimer_active(&rq->hrtick_timer))
		hrtimer_cancel(&rq->hrtick_timer);
}

/*
 * High-resolution timer tick.
 * Runs from hardirq context with interrupts disabled.
 */
/*
 * hrtick - 在硬中断上下文更新运行队列时钟，
 * 并执行调度类的高精度 tick 回调。
 * 定时器只触发一次，不自动重启。
 */
static enum hrtimer_restart hrtick(struct hrtimer *timer)
{
	struct rq *rq = container_of(timer, struct rq, hrtick_timer);
	struct rq_flags rf;

	WARN_ON_ONCE(cpu_of(rq) != smp_processor_id());

	rq_lock(rq, &rf);
	update_rq_clock(rq);
	/*
	 * queued=1 表示高精度时间片边界，而非普通周期 tick；调度类据此
	 * 判断是否请求抢占。这里只记账/设标志，硬中断中不直接换任务。
	 */
	rq->donor->sched_class->task_tick(rq, rq->curr, 1);
	rq_unlock(rq, &rf);

	return HRTIMER_NORESTART;
}

/*
 * __hrtick_restart - 按已保存的绝对到期时间启动
 * 绑定本 CPU 的高精度调度定时器。
 */
static void __hrtick_restart(struct rq *rq)
{
	struct hrtimer *timer = &rq->hrtick_timer;
	ktime_t time = rq->hrtick_time;

	hrtimer_start(timer, time, HRTIMER_MODE_ABS_PINNED_HARD);
}

/*
 * called from hardirq (IPI) context
 */
/*
 * __hrtick_start - 在目标 CPU 的 IPI 上下文加锁并启动高精度调度定时器。
 */
static void __hrtick_start(void *arg)
{
	struct rq *rq = arg;
	struct rq_flags rf;

	rq_lock(rq, &rf);
	__hrtick_restart(rq);
	rq_unlock(rq, &rf);
}

/*
 * Called to set the hrtick timer state.
 *
 * called with rq->lock held and IRQs disabled
 */
/*
 * hrtick_start - 设置至少 10 微秒的调度定时器延迟，
 * 必要时通过 IPI 在远端启动。
 * 调用者须持有运行队列锁且已关闭中断；delay 的单位为纳秒。
 */
void hrtick_start(struct rq *rq, u64 delay)
{
	struct hrtimer *timer = &rq->hrtick_timer;
	s64 delta;

	/*
	 * Don't schedule slices shorter than 10000ns, that just
	 * doesn't make sense and can cause timer DoS.
	 */
	delta = max_t(s64, delay, 10000LL);
	rq->hrtick_time = ktime_add_ns(hrtimer_cb_get_time(timer), delta);

	if (rq == this_rq())
		__hrtick_restart(rq);
	else
		/* PINNED 定时器须由目标 CPU 启动，异步 CSD 转交这项操作。 */
		smp_call_function_single_async(cpu_of(rq), &rq->hrtick_csd);
}

/*
 * hrtick_rq_init - 初始化运行队列的高精度调度定时器及远程启动回调。
 */
static void hrtick_rq_init(struct rq *rq)
{
	INIT_CSD(&rq->hrtick_csd, __hrtick_start, rq);
	hrtimer_setup(&rq->hrtick_timer, hrtick, CLOCK_MONOTONIC, HRTIMER_MODE_REL_HARD);
}
#else /* !CONFIG_SCHED_HRTICK: */
/*
 * hrtick_clear - 未配置高精度调度 tick 时的空操作清理接口。
 */
static inline void hrtick_clear(struct rq *rq)
{
}

/*
 * hrtick_rq_init - 未配置高精度调度 tick 时的空操作初始化接口。
 */
static inline void hrtick_rq_init(struct rq *rq)
{
}
#endif /* !CONFIG_SCHED_HRTICK */

/*
 * try_cmpxchg based fetch_or() macro so it works for different integer types:
 */
#define fetch_or(ptr, mask)						\
	({								\
		typeof(ptr) _ptr = (ptr);				\
		typeof(mask) _mask = (mask);				\
		typeof(*_ptr) _val = *_ptr;				\
									\
		do {							\
		} while (!try_cmpxchg(_ptr, &_val, _val | _mask));	\
	_val;								\
})

#ifdef TIF_POLLING_NRFLAG
/*
 * Atomically set TIF_NEED_RESCHED and test for TIF_POLLING_NRFLAG,
 * this avoids any races wrt polling state changes and thereby avoids
 * spurious IPIs.
 */
/*
 * set_nr_and_not_polling - 原子设置重调度标志；
 * 目标未轮询时返回 true，表示需要通知。
 */
static inline bool set_nr_and_not_polling(struct thread_info *ti, int tif)
{
	return !(fetch_or(&ti->flags, 1 << tif) & _TIF_POLLING_NRFLAG);
}

/*
 * Atomically set TIF_NEED_RESCHED if TIF_POLLING_NRFLAG is set.
 *
 * If this returns true, then the idle task promises to call
 * sched_ttwu_pending() and reschedule soon.
 */
/*
 * set_nr_if_polling - 仅在空闲任务轮询时原子设置重调度标志，
 * 成功时可省去唤醒 IPI。
 */
static bool set_nr_if_polling(struct task_struct *p)
{
	struct thread_info *ti = task_thread_info(p);
	typeof(ti->flags) val = READ_ONCE(ti->flags);

	/*
	 * “观察轮询”和“设置请求”必须属于同一次原子状态转换。
	 * 若 idle 在二者之间退出轮询，CAS 会失败并重读，调用者才能
	 * 正确选择发送 IPI，避免目标已经睡下却无人把它唤醒。
	 */
	do {
		if (!(val & _TIF_POLLING_NRFLAG))
			return false;
		if (val & _TIF_NEED_RESCHED)
			return true;
	} while (!try_cmpxchg(&ti->flags, &val, val | _TIF_NEED_RESCHED));

	return true;
}

#else
/*
 * set_nr_and_not_polling - 不支持轮询标志时直接设置重调度标志，
 * 并要求发送通知。
 */
static inline bool set_nr_and_not_polling(struct thread_info *ti, int tif)
{
	set_ti_thread_flag(ti, tif);
	return true;
}

/*
 * set_nr_if_polling - 不支持轮询标志时返回 false，禁止省略唤醒通知。
 */
static inline bool set_nr_if_polling(struct task_struct *p)
{
	return false;
}
#endif

/*
 * __wake_q_add - 原子认领任务的唤醒节点并加入队列，已被排队时返回 false。
 *
 * wake_q 是调用者收集“稍后需要唤醒谁”的临时清单，不是 CPU 的 rq，
 * 也不是存放等待者的 wait_queue_head。典型用法是在锁内选中等待者，
 * 解锁后 wake_up_q() 再逐一取得调度器锁，缩短原锁的持有时间。
 * head 的串接由当前上下文负责；同一 task 的节点认领需要原子操作，
 * 因为其他 CPU 也可能同时请求唤醒该任务。
 */
static bool __wake_q_add(struct wake_q_head *head, struct task_struct *task)
{
	struct wake_q_node *node = &task->wake_q;

	/*
	 * Atomically grab the task, if ->wake_q is !nil already it means
	 * it's already queued (either by us or someone else) and will get the
	 * wakeup due to that.
	 *
	 * In order to ensure that a pending wakeup will observe our pending
	 * state, even in the failed case, an explicit smp_mb() must be used.
	 *
	 * next 同时充当“已入队”标记；末节点用非空的 WAKE_Q_TAIL，
	 * 否则它会被误认为尚未入队。CAS 失败时也不能省略前面的屏障：
	 * 唤醒虽由已有队列负责，仍须让其观察到本调用者发布的条件。
	 */
	smp_mb__before_atomic();
	if (unlikely(cmpxchg_relaxed(&node->next, NULL, WAKE_Q_TAIL)))
		return false;

	/*
	 * The head is context local, there can be no concurrency.
	 */
	*head->lastp = node;
	head->lastp = &node->next;
	return true;
}

/**
 * wake_q_add() - queue a wakeup for 'later' waking.
 * 将任务加入批量唤醒队列，并在首次入队时获取任务引用。
 * @head: the wake_q_head to add @task to
 * @task: the task to queue for 'later' wakeup
 *
 * Queue a task for later wakeup, most likely by the wake_up_q() call in the
 * same context, _HOWEVER_ this is not guaranteed, the wakeup can come
 * instantly.
 *
 * This function must be used as-if it were wake_up_process(); IOW the task
 * must be ready to be woken at this location.
 */
void wake_q_add(struct wake_q_head *head, struct task_struct *task)
{
	if (__wake_q_add(head, task))
		get_task_struct(task);
}

/**
 * wake_q_add_safe() - safely queue a wakeup for 'later' waking.
 * 接管调用者已有的任务引用；任务重复入队时释放该引用。
 * @head: the wake_q_head to add @task to
 * @task: the task to queue for 'later' wakeup
 *
 * Queue a task for later wakeup, most likely by the wake_up_q() call in the
 * same context, _HOWEVER_ this is not guaranteed, the wakeup can come
 * instantly.
 *
 * This function must be used as-if it were wake_up_process(); IOW the task
 * must be ready to be woken at this location.
 *
 * This function is essentially a task-safe equivalent to wake_q_add(). Callers
 * that already hold reference to @task can call the 'safe' version and trust
 * wake_q to do the right thing depending whether or not the @task is already
 * queued for wakeup.
 */
void wake_q_add_safe(struct wake_q_head *head, struct task_struct *task)
{
	/*
	 * safe 指已有的任务引用会被接管，不表示这里马上完成了唤醒。
	 * 成功时引用留给 wake_up_q() 释放，失败时本处就释放，二者只能
	 * 发生一次；调用者不能再把同一份引用 put 一次。
	 */
	if (!__wake_q_add(head, task))
		put_task_struct(task);
}

/*
 * wake_up_q - 逐个唤醒队列中的任务，清除排队标记并释放对应任务引用。
 */
void wake_up_q(struct wake_q_head *head)
{
	struct wake_q_node *node = head->first;

	while (node != WAKE_Q_TAIL) {
		struct task_struct *task;

		task = container_of(node, struct task_struct, wake_q);
		/*
		 * 先保存后继再释放当前节点的入队标志；标志清零后其他 CPU
		 * 就可能复用 task->wake_q，不能再依赖该节点中的 next。
		 */
		node = node->next;
		/* pairs with cmpxchg_relaxed() in __wake_q_add() */
		WRITE_ONCE(task->wake_q.next, NULL);
		/* Task can safely be re-inserted now. */

		/*
		 * wake_up_process() executes a full barrier, which pairs with
		 * the queueing in wake_q_add() so as not to miss wakeups.
		 */
		wake_up_process(task);
		put_task_struct(task);
	}
}

/*
 * resched_curr - mark rq's current task 'to be rescheduled now'.
 *
 * On UP this means the setting of the need_resched flag, on SMP it
 * might also involve a cross-CPU call to trigger the scheduler on
 * the target CPU.
 */
/*
 * __resched_curr - 设置当前任务的指定重调度标志，
 * 必要时向远端 CPU 发送 IPI。
 * 调用者须持有运行队列锁；空闲任务的延迟抢占会提升为立即抢占。
 */
static void __resched_curr(struct rq *rq, int tif)
{
	struct task_struct *curr = rq->curr;
	struct thread_info *cti = task_thread_info(curr);
	int cpu;

	lockdep_assert_rq_held(rq);

	/*
	 * Always immediately preempt the idle task; no point in delaying doing
	 * actual work.
	 */
	if (is_idle_task(curr) && tif == TIF_NEED_RESCHED_LAZY)
		tif = TIF_NEED_RESCHED;

	if (cti->flags & ((1 << tif) | _TIF_NEED_RESCHED))
		return;

	cpu = cpu_of(rq);

	/*
	 * 这里只发布调度请求，不在持有 rq 锁时调用 schedule()。
	 * 本 CPU 同时维护线程标志和立即抢占所需的本地抢占标志。
	 */
	trace_sched_set_need_resched_tp(curr, cpu, tif);
	if (cpu == smp_processor_id()) {
		set_ti_thread_flag(cti, tif);
		if (tif == TIF_NEED_RESCHED)
			set_preempt_need_resched();
		return;
	}

	/*
	 * 目标空闲任务若在轮询，会自行观察请求；否则立即重调度需要 IPI
	 * 提醒目标 CPU。延迟重调度只设标志，不在此主动发送 IPI。
	 */
	if (set_nr_and_not_polling(cti, tif)) {
		if (tif == TIF_NEED_RESCHED)
			smp_send_reschedule(cpu);
	} else {
		trace_sched_wake_idle_without_ipi(cpu);
	}
}

/*
 * __trace_set_need_resched - 为本 CPU 上的重调度标志设置操作发出跟踪事件。
 */
void __trace_set_need_resched(struct task_struct *curr, int tif)
{
	trace_sched_set_need_resched_tp(curr, smp_processor_id(), tif);
}
EXPORT_SYMBOL_GPL(__trace_set_need_resched);

/*
 * resched_curr - 在运行队列锁保护下请求当前任务尽快重新调度。
 */
void resched_curr(struct rq *rq)
{
	__resched_curr(rq, TIF_NEED_RESCHED);
}

#ifdef CONFIG_PREEMPT_DYNAMIC
static DEFINE_STATIC_KEY_FALSE(sk_dynamic_preempt_lazy);
/*
 * dynamic_preempt_lazy - 查询运行时是否启用了延迟抢占模式。
 */
static __always_inline bool dynamic_preempt_lazy(void)
{
	return static_branch_unlikely(&sk_dynamic_preempt_lazy);
}
#else
/*
 * dynamic_preempt_lazy - 不支持动态抢占时返回编译期的延迟抢占配置。
 */
static __always_inline bool dynamic_preempt_lazy(void)
{
	return IS_ENABLED(CONFIG_PREEMPT_LAZY);
}
#endif

/*
 * get_lazy_tif_bit - 根据当前抢占模式选择延迟或立即重调度标志位。
 */
static __always_inline int get_lazy_tif_bit(void)
{
	if (dynamic_preempt_lazy())
		return TIF_NEED_RESCHED_LAZY;

	return TIF_NEED_RESCHED;
}

/*
 * resched_curr_lazy - 按当前抢占模式为运行队列请求可延迟的重新调度。
 */
void resched_curr_lazy(struct rq *rq)
{
	__resched_curr(rq, get_lazy_tif_bit());
}

/*
 * resched_cpu - 加锁后请求指定在线 CPU 或本 CPU 重新调度。
 */
void resched_cpu(int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	unsigned long flags;

	raw_spin_rq_lock_irqsave(rq, flags);
	if (cpu_online(cpu) || cpu == smp_processor_id())
		resched_curr(rq);
	raw_spin_rq_unlock_irqrestore(rq, flags);
}

#ifdef CONFIG_NO_HZ_COMMON
/*
 * In the semi idle case, use the nearest busy CPU for migrating timers
 * from an idle CPU.  This is good for power-savings.
 *
 * We don't do similar optimization for completely idle system, as
 * selecting an idle CPU will add more delays to the timers than intended
 * (as that CPU's timer base may not be up to date wrt jiffies etc).
 */
/*
 * get_nohz_timer_target - 为 NOHZ 定时器选择邻近的忙碌 housekeeping CPU。
 * 无忙碌候选时退回可用的 housekeeping CPU，以兼顾功耗与定时准确性。
 */
int get_nohz_timer_target(void)
{
	int i, cpu = smp_processor_id(), default_cpu = -1;
	struct sched_domain *sd;
	const struct cpumask *hk_mask;

	if (housekeeping_cpu(cpu, HK_TYPE_KERNEL_NOISE)) {
		if (!idle_cpu(cpu))
			return cpu;
		default_cpu = cpu;
	}

	hk_mask = housekeeping_cpumask(HK_TYPE_KERNEL_NOISE);

	guard(rcu)();

	for_each_domain(cpu, sd) {
		for_each_cpu_and(i, sched_domain_span(sd), hk_mask) {
			if (cpu == i)
				continue;

			if (!idle_cpu(i))
				return i;
		}
	}

	if (default_cpu == -1)
		default_cpu = housekeeping_any_cpu(HK_TYPE_KERNEL_NOISE);

	return default_cpu;
}

/*
 * When add_timer_on() enqueues a timer into the timer wheel of an
 * idle CPU then this timer might expire before the next timer event
 * which is scheduled to wake up that CPU. In case of a completely
 * idle system the next event might even be infinite time into the
 * future. wake_up_idle_cpu() ensures that the CPU is woken up and
 * leaves the inner idle loop so the newly added timer is taken into
 * account when the CPU goes back to idle and evaluates the timer
 * wheel for the next timer event.
 */
/*
 * wake_up_idle_cpu - 使远端空闲 CPU 退出空闲循环，
 * 重新评估新增定时器的到期时间。
 */
static void wake_up_idle_cpu(int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	if (cpu == smp_processor_id())
		return;

	/*
	 * Set TIF_NEED_RESCHED and send an IPI if in the non-polling
	 * part of the idle loop. This forces an exit from the idle loop
	 * and a round trip to schedule(). Now this could be optimized
	 * because a simple new idle loop iteration is enough to
	 * re-evaluate the next tick. Provided some re-ordering of tick
	 * nohz functions that would need to follow TIF_NR_POLLING
	 * clearing:
	 *
	 * - On most architectures, a simple fetch_or on ti::flags with a
	 *   "0" value would be enough to know if an IPI needs to be sent.
	 *
	 * - x86 needs to perform a last need_resched() check between
	 *   monitor and mwait which doesn't take timers into account.
	 *   There a dedicated TIF_TIMER flag would be required to
	 *   fetch_or here and be checked along with TIF_NEED_RESCHED
	 *   before mwait().
	 *
	 * However, remote timer enqueue is not such a frequent event
	 * and testing of the above solutions didn't appear to report
	 * much benefits.
	 */
	if (set_nr_and_not_polling(task_thread_info(rq->idle), TIF_NEED_RESCHED))
		smp_send_reschedule(cpu);
	else
		trace_sched_wake_idle_without_ipi(cpu);
}

/*
 * wake_up_full_nohz_cpu - 唤醒全动态 tick CPU 重新评估 tick；
 * 离线 CPU 也视为已处理。
 * 返回 false 表示目标不是全动态 tick CPU，需采用普通空闲唤醒路径。
 */
static bool wake_up_full_nohz_cpu(int cpu)
{
	/*
	 * We just need the target to call irq_exit() and re-evaluate
	 * the next tick. The nohz full kick at least implies that.
	 * If needed we can still optimize that later with an
	 * empty IRQ.
	 */
	if (cpu_is_offline(cpu))
		return true;  /* Don't try to wake offline CPUs. */
	if (tick_nohz_full_cpu(cpu)) {
		if (cpu != smp_processor_id() ||
		    tick_nohz_tick_stopped())
			tick_nohz_full_kick_cpu(cpu);
		return true;
	}

	return false;
}

/*
 * Wake up the specified CPU.  If the CPU is going offline, it is the
 * caller's responsibility to deal with the lost wakeup, for example,
 * by hooking into the CPU_DEAD notifier like timers and hrtimers do.
 */
/*
 * wake_up_nohz_cpu - 根据目标 CPU 的 NOHZ 模式选择
 * 全动态 tick 或普通空闲唤醒方式。
 */
void wake_up_nohz_cpu(int cpu)
{
	if (!wake_up_full_nohz_cpu(cpu))
		wake_up_idle_cpu(cpu);
}

/*
 * nohz_csd_func - 消费 NOHZ 均衡请求，并在目标仍空闲时触发调度软中断。
 */
static void nohz_csd_func(void *info)
{
	struct rq *rq = info;
	int cpu = cpu_of(rq);
	unsigned int flags;

	/*
	 * Release the rq::nohz_csd.
	 */
	flags = atomic_fetch_andnot(NOHZ_KICK_MASK | NOHZ_NEWILB_KICK, nohz_flags(cpu));
	WARN_ON(!(flags & NOHZ_KICK_MASK));

	rq->idle_balance = idle_cpu(cpu);
	if (rq->idle_balance) {
		/* IPI 只转交请求，实际均衡留给 SCHED_SOFTIRQ 执行。 */
		rq->nohz_idle_balance = flags;
		__raise_softirq_irqoff(SCHED_SOFTIRQ);
	}
}

#endif /* CONFIG_NO_HZ_COMMON */

#ifdef CONFIG_NO_HZ_FULL
/*
 * __need_bw_check - 判断唯一在队的公平类任务是否需要进一步检查带宽约束。
 */
static inline bool __need_bw_check(struct rq *rq, struct task_struct *p)
{
	if (rq->nr_running != 1)
		return false;

	if (p->sched_class != &fair_sched_class)
		return false;

	if (!task_on_rq_queued(p))
		return false;

	return true;
}

/*
 * sched_can_stop_tick - 综合调度类、可运行任务数量和带宽限制，
 * 判断能否停止调度 tick。
 */
bool sched_can_stop_tick(struct rq *rq)
{
	int fifo_nr_running;

	/*
	 * 停 tick 的前提是当前调度策略不再需要周期性干预，不能只数
	 * rq->nr_running：DL 预算、RR 轮转和公平组的带宽都可能要求 tick。
	 */
	/* Deadline tasks, even if single, need the tick */
	if (rq->dl.dl_nr_running)
		return false;

	/*
	 * If there are more than one RR tasks, we need the tick to affect the
	 * actual RR behaviour.
	 */
	if (rq->rt.rr_nr_running) {
		if (rq->rt.rr_nr_running == 1)
			return true;
		else
			return false;
	}

	/*
	 * If there's no RR tasks, but FIFO tasks, we can skip the tick, no
	 * forced preemption between FIFO tasks.
	 */
	fifo_nr_running = rq->rt.rt_nr_running - rq->rt.rr_nr_running;
	if (fifo_nr_running)
		return true;

	/*
	 * If there are no DL,RR/FIFO tasks, there must only be CFS or SCX tasks
	 * left. For CFS, if there's more than one we need the tick for
	 * involuntary preemption. For SCX, ask.
	 */
	if (scx_enabled() && !scx_can_stop_tick(rq))
		return false;

	if (rq->cfs.h_nr_queued > 1)
		return false;

	/*
	 * If there is one task and it has CFS runtime bandwidth constraints
	 * and it's on the cpu now we don't want to stop the tick.
	 * This check prevents clearing the bit if a newly enqueued task here is
	 * dequeued by migrating while the constrained task continues to run.
	 * E.g. going from 2->1 without going through pick_next_task().
	 */
	if (__need_bw_check(rq, rq->curr)) {
		if (cfs_task_bw_constrained(rq->curr))
			return false;
	}

	return true;
}
#endif /* CONFIG_NO_HZ_FULL */

#if defined(CONFIG_RT_GROUP_SCHED) || defined(CONFIG_FAIR_GROUP_SCHED)
/*
 * Iterate task_group tree rooted at *from, calling @down when first entering a
 * node and @up when leaving it for the final time.
 *
 * Caller must hold rcu_lock or sufficient equivalent.
 */
/*
 * walk_tg_tree_from - 深度优先遍历指定任务组子树，
 * 进入和离开节点时分别调用回调。
 * 调用者须提供 RCU 或等效保护；回调返回非零时终止遍历并返回该值。
 */
int walk_tg_tree_from(struct task_group *from,
			     tg_visitor down, tg_visitor up, void *data)
{
	struct task_group *parent, *child;
	int ret;

	parent = from;

down:
	ret = (*down)(parent, data);
	if (ret)
		goto out;
	list_for_each_entry_rcu(child, &parent->children, siblings) {
		parent = child;
		goto down;

up:
		continue;
	}
	ret = (*up)(parent, data);
	if (ret || parent == from)
		goto out;

	child = parent;
	parent = parent->parent;
	if (parent)
		goto up;
out:
	return ret;
}

/*
 * tg_nop - 任务组遍历使用的空回调，始终返回 0 以继续遍历。
 */
int tg_nop(struct task_group *tg, void *data)
{
	return 0;
}
#endif

/*
 * set_load_weight - 按静态优先级或空闲策略设置任务负载权重及其倒数。
 * update_load 为真且调度类支持时，同步更新已计入队列的负载。
 */
void set_load_weight(struct task_struct *p, bool update_load)
{
	int prio = p->static_prio - MAX_RT_PRIO;
	struct load_weight lw;

	if (task_has_idle_policy(p)) {
		lw.weight = scale_load(WEIGHT_IDLEPRIO);
		lw.inv_weight = WMULT_IDLEPRIO;
	} else {
		lw.weight = scale_load(sched_prio_to_weight[prio]);
		lw.inv_weight = sched_prio_to_wmult[prio];
	}

	/*
	 * SCHED_OTHER tasks have to update their load when changing their
	 * weight
	 */
	if (update_load && p->sched_class->reweight_task)
		p->sched_class->reweight_task(task_rq(p), p, &lw);
	else
		p->se.load = lw;
}

#ifdef CONFIG_UCLAMP_TASK
/*
 * Serializes updates of utilization clamp values
 *
 * The (slow-path) user-space triggers utilization clamp value updates which
 * can require updates on (fast-path) scheduler's data structures used to
 * support enqueue/dequeue operations.
 * While the per-CPU rq lock protects fast-path update operations, user-space
 * requests are serialized using a mutex to reduce the risk of conflicting
 * updates or API abuses.
 */
static __maybe_unused DEFINE_MUTEX(uclamp_mutex);

/*
 * uclamp 的三层表示：
 *   p->uclamp_req[]：任务请求，可能由用户或默认策略设置；
 *   p->uclamp[]：叠加组/系统约束后的有效值及所在桶的活动记录；
 *   rq->uclamp[]：本 CPU 所有参与计数任务的桶和聚合值。
 * 值使用 SCHED_CAPACITY_SCALE 的容量刻度，不是纳秒、时间片或带宽配额。
 * 它约束利用率信号对选核/频率决策的影响，不保证任务获得的 CPU 时间。
 */

/* Max allowed minimum utilization */
static unsigned int __maybe_unused sysctl_sched_uclamp_util_min = SCHED_CAPACITY_SCALE;

/* Max allowed maximum utilization */
static unsigned int __maybe_unused sysctl_sched_uclamp_util_max = SCHED_CAPACITY_SCALE;

/*
 * By default RT tasks run at the maximum performance point/capacity of the
 * system. Uclamp enforces this by always setting UCLAMP_MIN of RT tasks to
 * SCHED_CAPACITY_SCALE.
 *
 * This knob allows admins to change the default behavior when uclamp is being
 * used. In battery powered devices, particularly, running at the maximum
 * capacity and frequency will increase energy consumption and shorten the
 * battery life.
 *
 * This knob only affects RT tasks that their uclamp_se->user_defined == false.
 *
 * This knob will not override the system default sched_util_clamp_min defined
 * above.
 */
unsigned int sysctl_sched_uclamp_util_min_rt_default = SCHED_CAPACITY_SCALE;

/* All clamps are required to be less or equal than these values */
static struct uclamp_se uclamp_default[UCLAMP_CNT];

/*
 * This static key is used to reduce the uclamp overhead in the fast path. It
 * primarily disables the call to uclamp_rq_{inc, dec}() in
 * enqueue/dequeue_task().
 *
 * This allows users to continue to enable uclamp in their kernel config with
 * minimum uclamp overhead in the fast path.
 *
 * As soon as userspace modifies any of the uclamp knobs, the static key is
 * enabled, since we have an actual users that make use of uclamp
 * functionality.
 *
 * The knobs that would enable this static key are:
 *
 *   * A task modifying its uclamp value with sched_setattr().
 *   * An admin modifying the sysctl_sched_uclamp_{min, max} via procfs.
 *   * An admin modifying the cgroup cpu.uclamp.{min, max}
 */
DEFINE_STATIC_KEY_FALSE(sched_uclamp_used);

/*
 * uclamp_idle_value - 计算空闲队列的利用率钳制值，
 * 保留最后的最大钳制以避免频率抬升。
 */
static inline unsigned int
uclamp_idle_value(struct rq *rq, enum uclamp_id clamp_id,
		  unsigned int clamp_value)
{
	/*
	 * Avoid blocked utilization pushing up the frequency when we go
	 * idle (which drops the max-clamp) by retaining the last known
	 * max-clamp.
	 */
	if (clamp_id == UCLAMP_MAX) {
		rq->uclamp_flags |= UCLAMP_FLAG_IDLE;
		return clamp_value;
	}

	return uclamp_none(UCLAMP_MIN);
}

/*
 * uclamp_idle_reset - 队列退出空闲时用新任务的有效值重置保留的钳制值。
 */
static inline void uclamp_idle_reset(struct rq *rq, enum uclamp_id clamp_id,
				     unsigned int clamp_value)
{
	/* Reset max-clamp retention only on idle exit */
	if (!(rq->uclamp_flags & UCLAMP_FLAG_IDLE))
		return;

	uclamp_rq_set(rq, clamp_id, clamp_value);
}

/*
 * uclamp_rq_max_value - 从最高非空桶取得队列聚合钳制值，
 * 无任务时使用空闲默认值。
 */
static inline
unsigned int uclamp_rq_max_value(struct rq *rq, enum uclamp_id clamp_id,
				   unsigned int clamp_value)
{
	struct uclamp_bucket *bucket = rq->uclamp[clamp_id].bucket;
	int bucket_id = UCLAMP_BUCKETS - 1;

	/*
	 * Since both min and max clamps are max aggregated, find the
	 * top most bucket with tasks in.
	 *
	 * 这里不是把所有任务的 [min, max] 求交集：两类值都取最大值。
	 * min 满足最高的性能需求，max 则避免一个低上限任务压低同队列
	 * 其他任务的性能，因此从高桶向低桶查找即可。
	 */
	for ( ; bucket_id >= 0; bucket_id--) {
		if (!bucket[bucket_id].tasks)
			continue;
		return bucket[bucket_id].value;
	}

	/* No tasks -- default clamp values */
	return uclamp_idle_value(rq, clamp_id, clamp_value);
}

/*
 * __uclamp_update_util_min_rt_default - 更新实时任务的默认最小利用率。
 * 仅更新未被用户覆盖的请求。
 * 调用者须持有任务 pi_lock。
 */
static void __uclamp_update_util_min_rt_default(struct task_struct *p)
{
	unsigned int default_util_min;
	struct uclamp_se *uc_se;

	lockdep_assert_held(&p->pi_lock);

	uc_se = &p->uclamp_req[UCLAMP_MIN];

	/* Only sync if user didn't override the default */
	if (uc_se->user_defined)
		return;

	default_util_min = sysctl_sched_uclamp_util_min_rt_default;
	uclamp_se_set(uc_se, default_util_min, false);
}

/*
 * uclamp_update_util_min_rt_default - 锁定实时任务及其运行队列，
 * 同步默认最小利用率。
 */
static void uclamp_update_util_min_rt_default(struct task_struct *p)
{
	if (!rt_task(p))
		return;

	/* Protect updates to p->uclamp_* */
	guard(task_rq_lock)(p);
	__uclamp_update_util_min_rt_default(p);
}

/*
 * uclamp_tg_restrict - 将任务请求限制在所属任务组的有效利用率范围内。
 */
static inline struct uclamp_se
uclamp_tg_restrict(struct task_struct *p, enum uclamp_id clamp_id)
{
	/* Copy by value as we could modify it */
	struct uclamp_se uc_req = p->uclamp_req[clamp_id];
#ifdef CONFIG_UCLAMP_TASK_GROUP
	unsigned int tg_min, tg_max, value;

	/*
	 * Tasks in autogroups or root task group will be
	 * restricted by system defaults.
	 */
	if (task_group_is_autogroup(task_group(p)))
		return uc_req;
	if (task_group(p) == &root_task_group)
		return uc_req;

	tg_min = task_group(p)->uclamp[UCLAMP_MIN].value;
	tg_max = task_group(p)->uclamp[UCLAMP_MAX].value;
	value = uc_req.value;
	value = clamp(value, tg_min, tg_max);
	uclamp_se_set(&uc_req, value, false);
#endif

	return uc_req;
}

/*
 * The effective clamp bucket index of a task depends on, by increasing
 * priority:
 * - the task specific clamp value, when explicitly requested from userspace
 * - the task group effective clamp value, for tasks not either in the root
 *   group or in an autogroup
 * - the system default clamp value, defined by the sysadmin
 */
/*
 * uclamp_eff_get - 综合任务请求、任务组限制和系统上限，
 * 计算有效利用率钳制实体。
 */
static inline struct uclamp_se
uclamp_eff_get(struct task_struct *p, enum uclamp_id clamp_id)
{
	struct uclamp_se uc_req = uclamp_tg_restrict(p, clamp_id);
	struct uclamp_se uc_max = uclamp_default[clamp_id];

	/*
	 * 名字中的 default 还承担管理员上限的角色，并非仅在任务没有
	 * 请求时才生效。分别处理 MIN/MAX，每个维度都先受组约束，再
	 * 受系统允许值限制；不能直接把 uclamp_req 当成最终生效值。
	 */
	/* System default restrictions always apply */
	if (unlikely(uc_req.value > uc_max.value))
		return uc_max;

	return uc_req;
}

/*
 * uclamp_eff_value - 返回任务有效钳制值；已参与队列计数时优先使用缓存值。
 */
unsigned long uclamp_eff_value(struct task_struct *p, enum uclamp_id clamp_id)
{
	struct uclamp_se uc_eff;

	/* Task currently refcounted: use back-annotated (effective) value */
	if (p->uclamp[clamp_id].active)
		return (unsigned long)p->uclamp[clamp_id].value;

	uc_eff = uclamp_eff_get(p, clamp_id);

	return (unsigned long)uc_eff.value;
}

/*
 * When a task is enqueued on a rq, the clamp bucket currently defined by the
 * task's uclamp::bucket_id is refcounted on that rq. This also immediately
 * updates the rq's clamp value if required.
 *
 * Tasks can have a task-specific value requested from user-space, track
 * within each bucket the maximum value for tasks refcounted in it.
 * This "local max aggregation" allows to track the exact "requested" value
 * for each bucket when all its RUNNABLE tasks require the same clamp.
 */
/*
 * uclamp_rq_inc_id - 为单类钳制增加任务桶引用，
 * 并更新桶内及队列的最大聚合值。
 * 调用者须持有运行队列锁。
 */
static inline void uclamp_rq_inc_id(struct rq *rq, struct task_struct *p,
				    enum uclamp_id clamp_id)
{
	struct uclamp_rq *uc_rq = &rq->uclamp[clamp_id];
	struct uclamp_se *uc_se = &p->uclamp[clamp_id];
	struct uclamp_bucket *bucket;

	lockdep_assert_rq_held(rq);

	/* Update task effective clamp */
	p->uclamp[clamp_id] = uclamp_eff_get(p, clamp_id);

	/*
	 * 记录本次入队实际计入的桶；出队必须按这份有效值减计数，
	 * 不能根据期间可能被修改的 uclamp_req 再算一个桶。
	 */
	bucket = &uc_rq->bucket[uc_se->bucket_id];
	bucket->tasks++;
	uc_se->active = true;

	uclamp_idle_reset(rq, clamp_id, uc_se->value);

	/*
	 * Local max aggregation: rq buckets always track the max
	 * "requested" clamp value of its RUNNABLE tasks.
	 */
	if (bucket->tasks == 1 || uc_se->value > bucket->value)
		bucket->value = uc_se->value;

	if (uc_se->value > uclamp_rq_get(rq, clamp_id))
		uclamp_rq_set(rq, clamp_id, uc_se->value);
}

/*
 * When a task is dequeued from a rq, the clamp bucket refcounted by the task
 * is released. If this is the last task reference counting the rq's max
 * active clamp value, then the rq's clamp value is updated.
 *
 * Both refcounted tasks and rq's cached clamp values are expected to be
 * always valid. If it's detected they are not, as defensive programming,
 * enforce the expected state and warn.
 */
/*
 * uclamp_rq_dec_id - 释放单类钳制的任务桶引用，必要时重新计算队列有效值。
 * 调用者须持有运行队列锁，并允许跳过尚未参与计数的任务。
 */
static inline void uclamp_rq_dec_id(struct rq *rq, struct task_struct *p,
				    enum uclamp_id clamp_id)
{
	struct uclamp_rq *uc_rq = &rq->uclamp[clamp_id];
	struct uclamp_se *uc_se = &p->uclamp[clamp_id];
	struct uclamp_bucket *bucket;
	unsigned int bkt_clamp;
	unsigned int rq_clamp;

	lockdep_assert_rq_held(rq);

	/*
	 * If sched_uclamp_used was enabled after task @p was enqueued,
	 * we could end up with unbalanced call to uclamp_rq_dec_id().
	 *
	 * In this case the uc_se->active flag should be false since no uclamp
	 * accounting was performed at enqueue time and we can just return
	 * here.
	 *
	 * Need to be careful of the following enqueue/dequeue ordering
	 * problem too
	 *
	 *	enqueue(taskA)
	 *	// sched_uclamp_used gets enabled
	 *	enqueue(taskB)
	 *	dequeue(taskA)
	 *	// Must not decrement bucket->tasks here
	 *	dequeue(taskB)
	 *
	 * where we could end up with stale data in uc_se and
	 * bucket[uc_se->bucket_id].
	 *
	 * The following check here eliminates the possibility of such race.
	 */
	if (unlikely(!uc_se->active))
		return;

	bucket = &uc_rq->bucket[uc_se->bucket_id];

	WARN_ON_ONCE(!bucket->tasks);
	if (likely(bucket->tasks))
		bucket->tasks--;

	uc_se->active = false;

	/*
	 * Keep "local max aggregation" simple and accept to (possibly)
	 * overboost some RUNNABLE tasks in the same bucket.
	 * The rq clamp bucket value is reset to its base value whenever
	 * there are no more RUNNABLE tasks refcounting it.
	 *
	 * 最大值对应的任务离开后，只要桶未空就允许缓存值暂时偏大，
	 * 用少量过度提升换取出队时不必遍历桶内全部任务。
	 */
	if (likely(bucket->tasks))
		return;

	rq_clamp = uclamp_rq_get(rq, clamp_id);
	/*
	 * Defensive programming: this should never happen. If it happens,
	 * e.g. due to future modification, warn and fix up the expected value.
	 */
	WARN_ON_ONCE(bucket->value > rq_clamp);
	if (bucket->value >= rq_clamp) {
		bkt_clamp = uclamp_rq_max_value(rq, clamp_id, uc_se->value);
		uclamp_rq_set(rq, clamp_id, bkt_clamp);
	}
}

/*
 * uclamp_rq_inc - 将符合条件的入队任务计入各类利用率钳制桶，
 * 并清除空闲保留状态。
 */
static inline void uclamp_rq_inc(struct rq *rq, struct task_struct *p, int flags)
{
	enum uclamp_id clamp_id;

	/*
	 * Avoid any overhead until uclamp is actually used by the userspace.
	 *
	 * The condition is constructed such that a NOP is generated when
	 * sched_uclamp_used is disabled.
	 */
	if (!uclamp_is_used())
		return;

	if (unlikely(!p->sched_class->uclamp_enabled))
		return;

	/* Only inc the delayed task which being woken up. */
	if (p->se.sched_delayed && !(flags & ENQUEUE_DELAYED))
		return;

	for_each_clamp_id(clamp_id)
		uclamp_rq_inc_id(rq, p, clamp_id);

	/* Reset clamp idle holding when there is one RUNNABLE task */
	if (rq->uclamp_flags & UCLAMP_FLAG_IDLE)
		rq->uclamp_flags &= ~UCLAMP_FLAG_IDLE;
}

/*
 * uclamp_rq_dec - 将正常出队任务从各类利用率钳制桶中移除。
 */
static inline void uclamp_rq_dec(struct rq *rq, struct task_struct *p)
{
	enum uclamp_id clamp_id;

	/*
	 * Avoid any overhead until uclamp is actually used by the userspace.
	 *
	 * The condition is constructed such that a NOP is generated when
	 * sched_uclamp_used is disabled.
	 */
	if (!uclamp_is_used())
		return;

	if (unlikely(!p->sched_class->uclamp_enabled))
		return;

	if (p->se.sched_delayed)
		return;

	for_each_clamp_id(clamp_id)
		uclamp_rq_dec_id(rq, p, clamp_id);
}

/*
 * uclamp_rq_reinc_id - 对已活动的任务重新计数，以应用更新后的有效钳制值。
 */
static inline void uclamp_rq_reinc_id(struct rq *rq, struct task_struct *p,
				      enum uclamp_id clamp_id)
{
	if (!p->uclamp[clamp_id].active)
		return;

	/* 同一 rq 锁内先撤销旧桶，再按新约束计入新桶，保持引用成对。 */
	uclamp_rq_dec_id(rq, p, clamp_id);
	uclamp_rq_inc_id(rq, p, clamp_id);

	/*
	 * Make sure to clear the idle flag if we've transiently reached 0
	 * active tasks on rq.
	 */
	if (clamp_id == UCLAMP_MAX && (rq->uclamp_flags & UCLAMP_FLAG_IDLE))
		rq->uclamp_flags &= ~UCLAMP_FLAG_IDLE;
}

/*
 * uclamp_update_active - 锁定任务和队列，
 * 重新应用其已活动的利用率钳制请求。
 */
static inline void
uclamp_update_active(struct task_struct *p)
{
	enum uclamp_id clamp_id;
	struct rq_flags rf;
	struct rq *rq;

	/*
	 * Lock the task and the rq where the task is (or was) queued.
	 *
	 * We might lock the (previous) rq of a !RUNNABLE task, but that's the
	 * price to pay to safely serialize util_{min,max} updates with
	 * enqueues, dequeues and migration operations.
	 * This is the same locking schema used by __set_cpus_allowed_ptr().
	 */
	rq = task_rq_lock(p, &rf);

	/*
	 * Setting the clamp bucket is serialized by task_rq_lock().
	 * If the task is not yet RUNNABLE and its task_struct is not
	 * affecting a valid clamp bucket, the next time it's enqueued,
	 * it will already see the updated clamp bucket value.
	 */
	for_each_clamp_id(clamp_id)
		uclamp_rq_reinc_id(rq, p, clamp_id);

	task_rq_unlock(rq, p, &rf);
}

#ifdef CONFIG_UCLAMP_TASK_GROUP
/*
 * uclamp_update_active_tasks - 遍历指定 cgroup 的任务，刷新其活动钳制计数。
 */
static inline void
uclamp_update_active_tasks(struct cgroup_subsys_state *css)
{
	struct css_task_iter it;
	struct task_struct *p;

	css_task_iter_start(css, 0, &it);
	while ((p = css_task_iter_next(&it)))
		uclamp_update_active(p);
	css_task_iter_end(&it);
}

static void cpu_util_update_eff(struct cgroup_subsys_state *css);
#endif

#ifdef CONFIG_SYSCTL
#ifdef CONFIG_UCLAMP_TASK_GROUP
/*
 * uclamp_update_root_tg - 将系统钳制上限同步到根任务组，
 * 并向子组传播有效值。
 */
static void uclamp_update_root_tg(void)
{
	struct task_group *tg = &root_task_group;

	uclamp_se_set(&tg->uclamp_req[UCLAMP_MIN],
		      sysctl_sched_uclamp_util_min, false);
	uclamp_se_set(&tg->uclamp_req[UCLAMP_MAX],
		      sysctl_sched_uclamp_util_max, false);

	guard(rcu)();
	cpu_util_update_eff(&root_task_group.css);
}
#else
/*
 * uclamp_update_root_tg - 未启用任务组钳制时的空操作同步接口。
 */
static void uclamp_update_root_tg(void) { }
#endif

/*
 * uclamp_sync_util_min_rt_default - 与 fork 路径同步后，
 * 为所有实时任务更新默认下限。
 */
static void uclamp_sync_util_min_rt_default(void)
{
	struct task_struct *g, *p;

	/*
	 * copy_process()			sysctl_uclamp
	 *					  uclamp_min_rt = X;
	 *   write_lock(&tasklist_lock)		  read_lock(&tasklist_lock)
	 *   // link thread			  smp_mb__after_spinlock()
	 *   write_unlock(&tasklist_lock)	  read_unlock(&tasklist_lock);
	 *   sched_post_fork()			  for_each_process_thread()
	 *     __uclamp_sync_rt()		    __uclamp_sync_rt()
	 *
	 * Ensures that either sched_post_fork() will observe the new
	 * uclamp_min_rt or for_each_process_thread() will observe the new
	 * task.
	 */
	read_lock(&tasklist_lock);
	smp_mb__after_spinlock();
	read_unlock(&tasklist_lock);

	guard(rcu)();
	for_each_process_thread(g, p)
		uclamp_update_util_min_rt_default(p);
}

/*
 * sysctl_sched_uclamp_handler - 处理系统利用率钳制的 sysctl 读写，
 * 校验并传播配置。
 * 更新失败时恢复旧值，成功返回 0，否则返回错误码。
 */
static int sysctl_sched_uclamp_handler(const struct ctl_table *table, int write,
				void *buffer, size_t *lenp, loff_t *ppos)
{
	bool update_root_tg = false;
	int old_min, old_max, old_min_rt;
	int result;

	guard(mutex)(&uclamp_mutex);

	old_min = sysctl_sched_uclamp_util_min;
	old_max = sysctl_sched_uclamp_util_max;
	old_min_rt = sysctl_sched_uclamp_util_min_rt_default;

	result = proc_dointvec(table, write, buffer, lenp, ppos);
	if (result)
		goto undo;
	if (!write)
		return 0;

	if (sysctl_sched_uclamp_util_min > sysctl_sched_uclamp_util_max ||
	    sysctl_sched_uclamp_util_max > SCHED_CAPACITY_SCALE	||
	    sysctl_sched_uclamp_util_min_rt_default > SCHED_CAPACITY_SCALE) {

		result = -EINVAL;
		goto undo;
	}

	if (old_min != sysctl_sched_uclamp_util_min) {
		uclamp_se_set(&uclamp_default[UCLAMP_MIN],
			      sysctl_sched_uclamp_util_min, false);
		update_root_tg = true;
	}
	if (old_max != sysctl_sched_uclamp_util_max) {
		uclamp_se_set(&uclamp_default[UCLAMP_MAX],
			      sysctl_sched_uclamp_util_max, false);
		update_root_tg = true;
	}

	if (update_root_tg) {
		sched_uclamp_enable();
		uclamp_update_root_tg();
	}

	if (old_min_rt != sysctl_sched_uclamp_util_min_rt_default) {
		sched_uclamp_enable();
		uclamp_sync_util_min_rt_default();
	}

	/*
	 * We update all RUNNABLE tasks only when task groups are in use.
	 * Otherwise, keep it simple and do just a lazy update at each next
	 * task enqueue time.
	 */
	return 0;

undo:
	sysctl_sched_uclamp_util_min = old_min;
	sysctl_sched_uclamp_util_max = old_max;
	sysctl_sched_uclamp_util_min_rt_default = old_min_rt;
	return result;
}
#endif /* CONFIG_SYSCTL */

/*
 * uclamp_fork - 初始化子任务的钳制活动状态，
 * 并按 reset-on-fork 要求重置请求。
 */
static void uclamp_fork(struct task_struct *p)
{
	enum uclamp_id clamp_id;

	/*
	 * We don't need to hold task_rq_lock() when updating p->uclamp_* here
	 * as the task is still at its early fork stages.
	 */
	for_each_clamp_id(clamp_id)
		p->uclamp[clamp_id].active = false;

	if (likely(!p->sched_reset_on_fork))
		return;

	for_each_clamp_id(clamp_id) {
		uclamp_se_set(&p->uclamp_req[clamp_id],
			      uclamp_none(clamp_id), false);
	}
}

/*
 * uclamp_post_fork - 在任务创建完成后同步实时任务默认最小利用率。
 */
static void uclamp_post_fork(struct task_struct *p)
{
	uclamp_update_util_min_rt_default(p);
}

/*
 * init_uclamp_rq - 初始化单个运行队列的钳制聚合值，并标记为空闲状态。
 */
static void __init init_uclamp_rq(struct rq *rq)
{
	enum uclamp_id clamp_id;
	struct uclamp_rq *uc_rq = rq->uclamp;

	for_each_clamp_id(clamp_id) {
		uc_rq[clamp_id] = (struct uclamp_rq) {
			.value = uclamp_none(clamp_id)
		};
	}

	rq->uclamp_flags = UCLAMP_FLAG_IDLE;
}

/*
 * init_uclamp - 初始化所有运行队列、初始任务及根任务组
 * 的利用率钳制默认值。
 */
static void __init init_uclamp(void)
{
	struct uclamp_se uc_max = {};
	enum uclamp_id clamp_id;
	int cpu;

	for_each_possible_cpu(cpu)
		init_uclamp_rq(cpu_rq(cpu));

	for_each_clamp_id(clamp_id) {
		uclamp_se_set(&init_task.uclamp_req[clamp_id],
			      uclamp_none(clamp_id), false);
	}

	/* System defaults allow max clamp values for both indexes */
	uclamp_se_set(&uc_max, uclamp_none(UCLAMP_MAX), false);
	for_each_clamp_id(clamp_id) {
		uclamp_default[clamp_id] = uc_max;
#ifdef CONFIG_UCLAMP_TASK_GROUP
		root_task_group.uclamp_req[clamp_id] = uc_max;
		root_task_group.uclamp[clamp_id] = uc_max;
#endif
	}
}

#else /* !CONFIG_UCLAMP_TASK: */
/*
 * uclamp_rq_inc - 未启用任务钳制时的空操作入队计数接口。
 */
static inline void uclamp_rq_inc(struct rq *rq, struct task_struct *p, int flags) { }
/*
 * uclamp_rq_dec - 未启用任务钳制时的空操作出队计数接口。
 */
static inline void uclamp_rq_dec(struct rq *rq, struct task_struct *p) { }
/*
 * uclamp_fork - 未启用任务钳制时的空操作 fork 初始化接口。
 */
static inline void uclamp_fork(struct task_struct *p) { }
/*
 * uclamp_post_fork - 未启用任务钳制时的空操作 fork 后处理接口。
 */
static inline void uclamp_post_fork(struct task_struct *p) { }
/*
 * init_uclamp - 未启用任务钳制时的空操作全局初始化接口。
 */
static inline void init_uclamp(void) { }
#endif /* !CONFIG_UCLAMP_TASK */

/*
 * sched_task_on_rq - 判断任务是否处于运行队列的已排队状态，
 * 不包含迁移中状态。
 */
bool sched_task_on_rq(struct task_struct *p)
{
	return task_on_rq_queued(p);
}

/*
 * get_wchan - 在任务保持阻塞时读取其等待位置地址；
 * 无法安全获取时返回 0。
 */
unsigned long get_wchan(struct task_struct *p)
{
	unsigned long ip = 0;
	unsigned int state;

	if (!p || p == current)
		return 0;

	/* Only get wchan if task is blocked and we can keep it that way. */
	raw_spin_lock_irq(&p->pi_lock);
	state = READ_ONCE(p->__state);
	smp_rmb(); /* see try_to_wake_up() */
	if (state != TASK_RUNNING && state != TASK_WAKING && !p->on_rq)
		ip = __get_wchan(p);
	raw_spin_unlock_irq(&p->pi_lock);

	return ip;
}

/*
 * enqueue_task - 调用调度类将任务入队，
 * 同时更新时钟、钳制、PSI 和核心调度信息。
 * 调用者须持有运行队列锁。
 */
/*
 * 队列操作分两层：enqueue/dequeue_task() 操作调度类及公共记账，
 * activate/deactivate_task() 还管理任务的全局 on_rq 状态。
 * 修改属性时用 SAVE/RESTORE 临时摘下再放回，不是一次新的睡眠/唤醒；
 * NOCLOCK 表示调用链上已更新时钟，避免重复读取和重复记账。
 *
 * 对照调用原因理解 flags，比记住每个数值更有用：
 *   SLEEP / WAKEUP：任务从可运行变成阻塞，或从阻塞恢复。
 *   SAVE / RESTORE：为了改属性暂时摘下，尽量保留已有等待统计。
 *   MOVE：与 SAVE/RESTORE 配合时不保留原队列位置。
 *   DELAYED：完成延迟出队，或撤销延迟出队以恢复可运行资格。
 * 注意：SLEEP 不保证立刻从公平队列移除，dequeue_task() 的布尔
 * 返回值正是为了把“接受睡眠请求”与“实体已经移除”区分开。
 */
void enqueue_task(struct rq *rq, struct task_struct *p, int flags)
{
	if (!(flags & ENQUEUE_NOCLOCK))
		update_rq_clock(rq);

	/*
	 * Can be before ->enqueue_task() because uclamp considers the
	 * ENQUEUE_DELAYED task before its ->sched_delayed gets cleared
	 * in ->enqueue_task().
	 */
	uclamp_rq_inc(rq, p, flags);

	p->sched_class->enqueue_task(rq, p, flags);

	psi_enqueue(p, flags);

	if (!(flags & ENQUEUE_RESTORE))
		sched_info_enqueue(rq, p);

	if (sched_core_enabled(rq))
		sched_core_enqueue(rq, p);
}

/*
 * Must only return false when DEQUEUE_SLEEP.
 */
/*
 * dequeue_task - 更新公共调度统计并调用调度类将任务出队。
 * 持有运行队列锁调用；仅睡眠出队可能返回 false，表示延迟实际移除。
 */
inline bool dequeue_task(struct rq *rq, struct task_struct *p, int flags)
{
	if (sched_core_enabled(rq))
		sched_core_dequeue(rq, p, flags);

	if (!(flags & DEQUEUE_NOCLOCK))
		update_rq_clock(rq);

	if (!(flags & DEQUEUE_SAVE))
		sched_info_dequeue(rq, p);

	psi_dequeue(p, flags);

	/*
	 * Must be before ->dequeue_task() because ->dequeue_task() can 'fail'
	 * and mark the task ->sched_delayed.
	 */
	uclamp_rq_dec(rq, p);
	return p->sched_class->dequeue_task(rq, p, flags);
}

/*
 * activate_task - 处理迁移相关状态并将任务入队，随后发布已排队标志。
 */
void activate_task(struct rq *rq, struct task_struct *p, int flags)
{
	if (task_on_rq_migrating(p))
		flags |= ENQUEUE_MIGRATED;
	if (flags & ENQUEUE_MIGRATED)
		sched_mm_cid_migrate_to(rq, p);

	enqueue_task(rq, p, flags);

	/* QUEUED 是入队完成后的发布点；半迁移期间仍保持 MIGRATING。 */
	WRITE_ONCE(p->on_rq, TASK_ON_RQ_QUEUED);
	ASSERT_EXCLUSIVE_WRITER(p->on_rq);
}

/*
 * deactivate_task - 将任务标记为迁移中后出队，供非睡眠的队列转移使用。
 */
void deactivate_task(struct rq *rq, struct task_struct *p, int flags)
{
	WARN_ON_ONCE(flags & DEQUEUE_SLEEP);

	WRITE_ONCE(p->on_rq, TASK_ON_RQ_MIGRATING);
	ASSERT_EXCLUSIVE_WRITER(p->on_rq);

	/*
	 * Code explicitly relies on TASK_ON_RQ_MIGRATING begin set *before*
	 * dequeue_task() and cleared *after* enqueue_task().
	 */

	dequeue_task(rq, p, flags);
}

/*
 * block_task - 尝试按睡眠原因出队，仅在调度类完成出队后执行阻塞记账。
 */
static void block_task(struct rq *rq, struct task_struct *p, int flags)
{
	/*
	 * 公平类可能只标记 sched_delayed，暂时保留队列中的调度实体。
	 * 只有真正完成出队后，才能由 __block_task() 清除 on_rq 并记账。
	 */
	if (dequeue_task(rq, p, DEQUEUE_SLEEP | flags))
		__block_task(rq, p);
}

/**
 * task_curr - is this task currently executing on a CPU?
 * 判断任务是否为其所属 CPU 的当前执行任务，结果为 1 或 0。
 * @p: the task in question.
 *
 * Return: 1 if the task is currently executing. 0 otherwise.
 */
inline int task_curr(const struct task_struct *p)
{
	return cpu_curr(task_cpu(p)) == p;
}

/*
 * ->switching_to() is called with the pi_lock and rq_lock held and must not
 * mess with locking.
 */
/*
 * check_class_changing - 调度类改变时调用新调度类的切入准备回调。
 * 调用者须持有 pi_lock 和运行队列锁，回调不得改变锁状态。
 */
void check_class_changing(struct rq *rq, struct task_struct *p,
			  const struct sched_class *prev_class)
{
	if (prev_class != p->sched_class && p->sched_class->switching_to)
		p->sched_class->switching_to(rq, p);
}

/*
 * switched_from, switched_to and prio_changed must _NOT_ drop rq->lock,
 * use the balance_callback list if you want balancing.
 *
 * this means any call to check_class_changed() must be followed by a call to
 * balance_callback().
 */
/*
 * check_class_changed - 根据调度类或优先级变化调用相应通知回调。
 * 回调不得释放运行队列锁；调用后还需处理均衡回调。
 */
void check_class_changed(struct rq *rq, struct task_struct *p,
			 const struct sched_class *prev_class,
			 int oldprio)
{
	if (prev_class != p->sched_class) {
		if (prev_class->switched_from)
			prev_class->switched_from(rq, p);

		p->sched_class->switched_to(rq, p);
	} else if (oldprio != p->prio || dl_task(p))
		p->sched_class->prio_changed(rq, p, oldprio);
}

/*
 * wakeup_preempt - 判断新唤醒任务能否抢占当前调度资源提供者，
 * 并请求重新调度。
 */
void wakeup_preempt(struct rq *rq, struct task_struct *p, int flags)
{
	struct task_struct *donor = rq->donor;

	/*
	 * 例如 RT 任务唤醒时可直接压过 fair 类；两个 fair 任务之间谁
	 * 更值得运行，则须由 fair 自己比较。这里请求抢占的是执行者
	 * rq->curr，比较的调度资格却属于 donor，代理执行时尤须区分。
	 */
	if (p->sched_class == donor->sched_class)
		donor->sched_class->wakeup_preempt(rq, p, flags);
	else if (sched_class_above(p->sched_class, donor->sched_class))
		resched_curr(rq);

	/*
	 * A queue event has occurred, and we're going to schedule.  In
	 * this case, we can save a useless back to back clock update.
	 */
	if (task_on_rq_queued(donor) && test_tsk_need_resched(rq->curr))
		rq_clock_skip_update(rq);
}

/*
 * __task_state_match - 匹配任务当前或保存状态，
 * 分别返回 1、-1；均不匹配时返回 0。
 */
static __always_inline
int __task_state_match(struct task_struct *p, unsigned int state)
{
	if (READ_ONCE(p->__state) & state)
		return 1;

	if (READ_ONCE(p->saved_state) & state)
		return -1;

	return 0;
}

/*
 * task_state_match - 在 pi_lock 保护下匹配任务状态，
 * 避免 RT 锁及冻结状态切换竞态。
 */
static __always_inline
int task_state_match(struct task_struct *p, unsigned int state)
{
	/*
	 * Serialize against current_save_and_set_rtlock_wait_state(),
	 * current_restore_rtlock_saved_state(), and __refrigerator().
	 */
	guard(raw_spinlock_irq)(&p->pi_lock);
	return __task_state_match(p, state);
}

/*
 * wait_task_inactive - wait for a thread to unschedule.
 *
 * Wait for the thread to block in any of the states set in @match_state.
 * If it changes, i.e. @p might have woken up, then return zero.  When we
 * succeed in waiting for @p to be off its CPU, we return a positive number
 * (its total switch count).  If a second call a short while later returns the
 * same number, the caller can be sure that @p has remained unscheduled the
 * whole time.
 *
 * The caller must ensure that the task *will* unschedule sometime soon,
 * else this function might spin for a *long* time. This function can't
 * be called with interrupts off, or it may introduce deadlock with
 * smp_call_function() if an IPI is sent by the same process we are
 * waiting to become inactive.
 */
/*
 * wait_task_inactive - 等待任务保持指定状态并完全离开 CPU 和运行队列。
 * 成功返回带最高位标记的切换计数，
 * 状态改变则返回 0；不可关中断调用。
 */
unsigned long wait_task_inactive(struct task_struct *p, unsigned int match_state)
{
	int running, queued, match;
	struct rq_flags rf;
	unsigned long ncsw;
	struct rq *rq;

	for (;;) {
		/*
		 * We do the initial early heuristics without holding
		 * any task-queue locks at all. We'll only try to get
		 * the runqueue lock when things look like they will
		 * work out!
		 */
		rq = task_rq(p);

		/*
		 * If the task is actively running on another CPU
		 * still, just relax and busy-wait without holding
		 * any locks.
		 *
		 * NOTE! Since we don't hold any locks, it's not
		 * even sure that "rq" stays as the right runqueue!
		 * But we don't care, since "task_on_cpu()" will
		 * return false if the runqueue has changed and p
		 * is actually now running somewhere else!
		 */
		while (task_on_cpu(rq, p)) {
			if (!task_state_match(p, match_state))
				return 0;
			cpu_relax();
		}

		/*
		 * Ok, time to look more closely! We need the rq
		 * lock now, to be *sure*. If we're wrong, we'll
		 * just go back and repeat.
		 */
		rq = task_rq_lock(p, &rf);
		/*
		 * If task is sched_delayed, force dequeue it, to avoid always
		 * hitting the tick timeout in the queued case
		 */
		if (p->se.sched_delayed)
			dequeue_task(rq, p, DEQUEUE_SLEEP | DEQUEUE_DELAYED);
		trace_sched_wait_task(p);
		running = task_on_cpu(rq, p);
		queued = task_on_rq_queued(p);
		ncsw = 0;
		if ((match = __task_state_match(p, match_state))) {
			/*
			 * When matching on p->saved_state, consider this task
			 * still queued so it will wait.
			 */
			if (match < 0)
				queued = 1;
			ncsw = p->nvcsw | LONG_MIN; /* sets MSB */
		}
		task_rq_unlock(rq, p, &rf);

		/*
		 * If it changed from the expected state, bail out now.
		 */
		if (unlikely(!ncsw))
			break;

		/*
		 * Was it really running after all now that we
		 * checked with the proper locks actually held?
		 *
		 * Oops. Go back and try again..
		 */
		if (unlikely(running)) {
			cpu_relax();
			continue;
		}

		/*
		 * It's not enough that it's not actively running,
		 * it must be off the runqueue _entirely_, and not
		 * preempted!
		 *
		 * So if it was still runnable (but just not actively
		 * running right now), it's preempted, and we should
		 * yield - it could be a while.
		 */
		if (unlikely(queued)) {
			ktime_t to = NSEC_PER_SEC / HZ;

			set_current_state(TASK_UNINTERRUPTIBLE);
			schedule_hrtimeout(&to, HRTIMER_MODE_REL_HARD);
			continue;
		}

		/*
		 * Ahh, all good. It wasn't running, and it wasn't
		 * runnable, which means that it will never become
		 * running in the future either. We're all done!
		 */
		break;
	}

	return ncsw;
}

static void
__do_set_cpus_allowed(struct task_struct *p, struct affinity_context *ctx);

/*
 * migrate_disable_switch - 切出禁止迁移的任务时，
 * 将其有效亲和性临时收缩至当前 CPU。
 */
static void migrate_disable_switch(struct rq *rq, struct task_struct *p)
{
	struct affinity_context ac = {
		.new_mask  = cpumask_of(rq->cpu),
		.flags     = SCA_MIGRATE_DISABLE,
	};

	/*
	 * migrate_disable() 允许任务被抢占，只禁止它换 CPU。直到真正
	 * 切出时才需把 cpus_ptr 指向单 CPU 掩码，让后续选核仍落在原处；
	 * cpus_mask 保留正常亲和性，migrate_enable() 再把指针切回。
	 */
	if (likely(!p->migration_disabled))
		return;

	if (p->cpus_ptr != &p->cpus_mask)
		return;

	/*
	 * Violates locking rules! See comment in __do_set_cpus_allowed().
	 */
	__do_set_cpus_allowed(p, &ac);
}

/*
 * ___migrate_enable - 恢复当前任务的正常亲和性，
 * 并落实禁止迁移期间积累的迁移请求。
 */
void ___migrate_enable(void)
{
	struct task_struct *p = current;
	struct affinity_context ac = {
		.new_mask  = &p->cpus_mask,
		.flags     = SCA_MIGRATE_ENABLE,
	};

	__set_cpus_allowed_ptr(p, &ac);
}
EXPORT_SYMBOL_GPL(___migrate_enable);

/*
 * migrate_disable - 导出禁止当前任务迁移的接口，允许嵌套调用。
 */
void migrate_disable(void)
{
	__migrate_disable();
}
EXPORT_SYMBOL_GPL(migrate_disable);

/*
 * migrate_enable - 与 migrate_disable 配对，在最外层退出时恢复任务迁移能力。
 */
void migrate_enable(void)
{
	__migrate_enable();
}
EXPORT_SYMBOL_GPL(migrate_enable);

/*
 * rq_has_pinned_tasks - 判断队列中是否存在因禁止迁移而固定在本 CPU 的任务。
 */
static inline bool rq_has_pinned_tasks(struct rq *rq)
{
	return rq->nr_pinned;
}

/*
 * Per-CPU kthreads are allowed to run on !active && online CPUs, see
 * __set_cpus_allowed_ptr() and select_fallback_rq().
 */
/*
 * is_cpu_allowed - 综合亲和性、迁移禁用状态和 CPU 热插拔状态，
 * 判断任务能否在此运行。
 */
static inline bool is_cpu_allowed(struct task_struct *p, int cpu)
{
	/*
	 * online 表示 CPU 已能参与运行，active 表示可承接普通任务放置。
	 * 热插拔过渡期两者不相同；绑定内核线程和禁止迁移任务需要
	 * 留在 online 但尚未/不再 active 的 CPU 上完成启动或收尾。
	 */
	/* When not in the task's cpumask, no point in looking further. */
	if (!task_allowed_on_cpu(p, cpu))
		return false;

	/* migrate_disabled() must be allowed to finish. */
	if (is_migration_disabled(p))
		return cpu_online(cpu);

	/* Non kernel threads are not allowed during either online or offline. */
	if (!(p->flags & PF_KTHREAD))
		return cpu_active(cpu);

	/* KTHREAD_IS_PER_CPU is always allowed. */
	if (kthread_is_per_cpu(p))
		return cpu_online(cpu);

	/* Regular kernel threads don't get to stay during offline. */
	if (cpu_dying(cpu))
		return false;

	/* But are allowed during online. */
	return cpu_online(cpu);
}

/*
 * This is how migration works:
 *
 * 1) we invoke migration_cpu_stop() on the target CPU using
 *    stop_one_cpu().
 * 2) stopper starts to run (implicitly forcing the migrated thread
 *    off the CPU)
 * 3) it checks whether the migrated task is still in the wrong runqueue.
 * 4) if it's in the wrong runqueue then the migration thread removes
 *    it and puts it into the right queue.
 * 5) stopper completes and stop_one_cpu() returns and the migration
 *    is done.
 */

/*
 * move_queued_task - move a queued task to new rq.
 *
 * Returns (locked) new rq. Old rq's lock is released.
 */
/*
 * move_queued_task - 将已排队任务迁移至目标 CPU，并检查目标上的抢占需求。
 * 返回已加锁的新队列，原队列锁在迁移过程中释放。
 */
static struct rq *move_queued_task(struct rq *rq, struct rq_flags *rf,
				   struct task_struct *p, int new_cpu)
{
	lockdep_assert_rq_held(rq);

	/*
	 * 不同时持有两个 rq 锁：先在源队列设置 MIGRATING，再改 CPU，
	 * 最后在目标队列入队时恢复 QUEUED。中间即使两把锁都未持有，
	 * task_rq_lock() 会因 MIGRATING 而重试，不会接受半迁移状态；
	 * 先释放源锁、再获取目标锁，还串起了迁移前后的内存序。
	 */
	deactivate_task(rq, p, DEQUEUE_NOCLOCK);
	set_task_cpu(p, new_cpu);
	rq_unlock(rq, rf);

	rq = cpu_rq(new_cpu);

	rq_lock(rq, rf);
	WARN_ON_ONCE(task_cpu(p) != new_cpu);
	activate_task(rq, p, 0);
	wakeup_preempt(rq, p, 0);

	return rq;
}

struct migration_arg {
	struct task_struct		*task;
	int				dest_cpu;
	struct set_affinity_pending	*pending;
};

/*
 * @refs: number of wait_for_completion()
 * @stop_pending: is @stop_work in use
 */
struct set_affinity_pending {
	/*
	 * 此对象可由首个请求者放在栈上：refs 保证所有等待者退出后
	 * 才能销毁，done 通知迁移要求已满足，stop_pending 保证同一
	 * stop_work 不被重复提交。三者分别管理生命周期、结果和在途工作。
	 */
	refcount_t		refs;
	unsigned int		stop_pending;
	struct completion	done;
	struct cpu_stop_work	stop_work;
	struct migration_arg	arg;
};

/*
 * Move (not current) task off this CPU, onto the destination CPU. We're doing
 * this because either it can't run here any more (set_cpus_allowed()
 * away from this CPU, or CPU going down), or because we're
 * attempting to rebalance this task on exec (sched_exec).
 *
 * So we race with normal scheduler movements, but that's OK, as long
 * as the task is no longer on this CPU.
 */
/*
 * __migrate_task - 验证目标 CPU 仍允许任务运行，再迁移非当前的已排队任务。
 * 返回保持加锁的最终运行队列；目标不合法时保留原队列。
 */
static struct rq *__migrate_task(struct rq *rq, struct rq_flags *rf,
				 struct task_struct *p, int dest_cpu)
{
	/* Affinity changed (again). */
	if (!is_cpu_allowed(p, dest_cpu))
		return rq;

	rq = move_queued_task(rq, rf, p, dest_cpu);

	return rq;
}

/*
 * migration_cpu_stop - this will be executed by a high-prio stopper thread
 * and performs thread migration by bumping thread off CPU then
 * 'pushing' onto another runqueue.
 */
/*
 * migration_cpu_stop - 在 stopper 上下文强制任务离开不允许的 CPU，
 * 并完成亲和性请求。
 * 处理任务已迁移、正在唤醒或暂时禁止迁移等并发情况。
 */
static int migration_cpu_stop(void *data)
{
	struct migration_arg *arg = data;
	struct set_affinity_pending *pending = arg->pending;
	struct task_struct *p = arg->task;
	struct rq *rq = this_rq();
	bool complete = false;
	struct rq_flags rf;

	/*
	 * The original target CPU might have gone down and we might
	 * be on another CPU but it doesn't matter.
	 */
	local_irq_save(rf.flags);
	/*
	 * We need to explicitly wake pending tasks before running
	 * __migrate_task() such that we will not miss enforcing cpus_ptr
	 * during wakeups, see set_cpus_allowed_ptr()'s TASK_WAKING test.
	 */
	flush_smp_call_function_queue();

	/*
	 * stopper 已把原执行任务挤下 CPU，再取得两层锁验证最新状态。
	 * 请求提交到执行之间可以发生唤醒、迁移、再次改亲和性；
	 * arg 中的 CPU 是请求时的意图，不能代替此处的重新检查。
	 */
	raw_spin_lock(&p->pi_lock);
	rq_lock(rq, &rf);

	/*
	 * If we were passed a pending, then ->stop_pending was set, thus
	 * p->migration_pending must have remained stable.
	 */
	WARN_ON_ONCE(pending && pending != p->migration_pending);

	/*
	 * If task_rq(p) != rq, it cannot be migrated here, because we're
	 * holding rq->lock, if p->on_rq == 0 it cannot get enqueued because
	 * we're holding p->pi_lock.
	 */
	if (task_rq(p) == rq) {
		if (is_migration_disabled(p))
			goto out;

		if (pending) {
			p->migration_pending = NULL;
			complete = true;

			if (cpumask_test_cpu(task_cpu(p), &p->cpus_mask))
				goto out;
		}

		if (task_on_rq_queued(p)) {
			update_rq_clock(rq);
			rq = __migrate_task(rq, &rf, p, arg->dest_cpu);
		} else {
			/* 睡眠任务只指定下次唤醒的首选 CPU，不搬队列。 */
			p->wake_cpu = arg->dest_cpu;
		}

		/*
		 * XXX __migrate_task() can fail, at which point we might end
		 * up running on a dodgy CPU, AFAICT this can only happen
		 * during CPU hotplug, at which point we'll get pushed out
		 * anyway, so it's probably not a big deal.
		 */

	} else if (pending) {
		/*
		 * This happens when we get migrated between migrate_enable()'s
		 * preempt_enable() and scheduling the stopper task. At that
		 * point we're a regular task again and not current anymore.
		 *
		 * A !PREEMPT kernel has a giant hole here, which makes it far
		 * more likely.
		 */

		/*
		 * The task moved before the stopper got to run. We're holding
		 * ->pi_lock, so the allowed mask is stable - if it got
		 * somewhere allowed, we're done.
		 */
		if (cpumask_test_cpu(task_cpu(p), p->cpus_ptr)) {
			p->migration_pending = NULL;
			complete = true;
			goto out;
		}

		/*
		 * When migrate_enable() hits a rq mis-match we can't reliably
		 * determine is_migration_disabled() and so have to chase after
		 * it.
		 */
		WARN_ON_ONCE(!pending->stop_pending);
		preempt_disable();
		task_rq_unlock(rq, p, &rf);
		stop_one_cpu_nowait(task_cpu(p), migration_cpu_stop,
				    &pending->arg, &pending->stop_work);
		preempt_enable();
		return 0;
	}
out:
	if (pending)
		pending->stop_pending = false;
	task_rq_unlock(rq, p, &rf);

	if (complete)
		complete_all(&pending->done);

	return 0;
}

/*
 * push_cpu_stop - 由 stopper 将任务推至调度类选择的队列，
 * 禁止迁移时记录待推送标志。
 */
int push_cpu_stop(void *arg)
{
	struct rq *lowest_rq = NULL, *rq = this_rq();
	struct task_struct *p = arg;

	raw_spin_lock_irq(&p->pi_lock);
	raw_spin_rq_lock(rq);

	if (task_rq(p) != rq)
		goto out_unlock;

	if (is_migration_disabled(p)) {
		p->migration_flags |= MDF_PUSH;
		goto out_unlock;
	}

	p->migration_flags &= ~MDF_PUSH;

	if (p->sched_class->find_lock_rq)
		lowest_rq = p->sched_class->find_lock_rq(p, rq);

	if (!lowest_rq)
		goto out_unlock;

	// XXX validate p is still the highest prio task
	if (task_rq(p) == rq) {
		move_queued_task_locked(rq, lowest_rq, p);
		resched_curr(lowest_rq);
	}

	double_unlock_balance(rq, lowest_rq);

out_unlock:
	rq->push_busy = false;
	raw_spin_rq_unlock(rq);
	raw_spin_unlock_irq(&p->pi_lock);

	put_task_struct(p);
	return 0;
}

/*
 * sched_class::set_cpus_allowed must do the below, but is not required to
 * actually call this function.
 */
/*
 * set_cpus_allowed_common - 更新通用亲和性字段，
 * 或按迁移禁用状态切换有效掩码指针。
 *
 * 亲和性有三个容易混淆的表示：
 *   cpus_mask：任务当前的正常亲和性集合；
 *   cpus_ptr：选核实际使用的集合，迁移禁用期间可指向单 CPU 掩码；
 *   user_cpus_ptr：按需保存的用户亲和性，用于约束调整后的恢复。
 * 改正常集合与临时固定 CPU 是不同操作；前者还要更新允许 CPU 数，
 * 后者只替换指针，避免把用户的正常集合破坏掉。
 */
void set_cpus_allowed_common(struct task_struct *p, struct affinity_context *ctx)
{
	if (ctx->flags & (SCA_MIGRATE_ENABLE | SCA_MIGRATE_DISABLE)) {
		p->cpus_ptr = ctx->new_mask;
		return;
	}

	cpumask_copy(&p->cpus_mask, ctx->new_mask);
	p->nr_cpus_allowed = cpumask_weight(ctx->new_mask);

	/*
	 * Swap in a new user_cpus_ptr if SCA_USER flag set
	 */
	if (ctx->flags & SCA_USER)
		swap(p->user_cpus_ptr, ctx->user_mask);
}

/*
 * __do_set_cpus_allowed - 临时保存任务队列状态，
 * 调用调度类更新亲和性后恢复入队和运行状态。
 * 通常要求持有 pi_lock；已排队任务还须持有运行队列锁。
 */
static void
__do_set_cpus_allowed(struct task_struct *p, struct affinity_context *ctx)
{
	struct rq *rq = task_rq(p);
	bool queued, running;

	/*
	 * This here violates the locking rules for affinity, since we're only
	 * supposed to change these variables while holding both rq->lock and
	 * p->pi_lock.
	 *
	 * HOWEVER, it magically works, because ttwu() is the only code that
	 * accesses these variables under p->pi_lock and only does so after
	 * smp_cond_load_acquire(&p->on_cpu, !VAL), and we're in __schedule()
	 * before finish_task().
	 *
	 * XXX do further audits, this smells like something putrid.
	 */
	if (ctx->flags & SCA_MIGRATE_DISABLE)
		WARN_ON_ONCE(!p->on_cpu);
	else
		lockdep_assert_held(&p->pi_lock);

	queued = task_on_rq_queued(p);
	running = task_current_donor(rq, p);

	/*
	 * 亲和性会影响调度类的可迁移任务索引和当前实体状态，不能只改
	 * cpumask 字段。先保存队列/当前身份，回调重建类内约束后再恢复；
	 * 这里只更新约束，是否需要换 CPU 由 affine_move_task() 决定。
	 */
	if (queued) {
		/*
		 * Because __kthread_bind() calls this on blocked tasks without
		 * holding rq->lock.
		 */
		lockdep_assert_rq_held(rq);
		dequeue_task(rq, p, DEQUEUE_SAVE | DEQUEUE_NOCLOCK);
	}
	if (running)
		put_prev_task(rq, p);

	p->sched_class->set_cpus_allowed(p, ctx);
	mm_set_cpus_allowed(p->mm, ctx->new_mask);

	if (queued)
		enqueue_task(rq, p, ENQUEUE_RESTORE | ENQUEUE_NOCLOCK);
	if (running)
		set_next_task(rq, p);
}

/*
 * Used for kthread_bind() and select_fallback_rq(), in both cases the user
 * affinity (if any) should be destroyed too.
 */
/*
 * do_set_cpus_allowed - 直接设置任务亲和性，
 * 并通过 RCU 释放被替换的用户亲和性掩码。
 */
void do_set_cpus_allowed(struct task_struct *p, const struct cpumask *new_mask)
{
	struct affinity_context ac = {
		.new_mask  = new_mask,
		.user_mask = NULL,
		.flags     = SCA_USER,	/* clear the user requested mask */
	};
	union cpumask_rcuhead {
		cpumask_t cpumask;
		struct rcu_head rcu;
	};

	__do_set_cpus_allowed(p, &ac);

	/*
	 * Because this is called with p->pi_lock held, it is not possible
	 * to use kfree() here (when PREEMPT_RT=y), therefore punt to using
	 * kfree_rcu().
	 */
	kfree_rcu((union cpumask_rcuhead *)ac.user_mask, rcu);
}

/*
 * dup_user_cpus_ptr - 为目标任务复制源任务的用户亲和性掩码，
 * 分配失败返回 -ENOMEM。
 */
int dup_user_cpus_ptr(struct task_struct *dst, struct task_struct *src,
		      int node)
{
	cpumask_t *user_mask;
	unsigned long flags;

	/*
	 * Always clear dst->user_cpus_ptr first as their user_cpus_ptr's
	 * may differ by now due to racing.
	 */
	dst->user_cpus_ptr = NULL;

	/*
	 * This check is racy and losing the race is a valid situation.
	 * It is not worth the extra overhead of taking the pi_lock on
	 * every fork/clone.
	 */
	if (data_race(!src->user_cpus_ptr))
		return 0;

	user_mask = alloc_user_cpus_ptr(node);
	if (!user_mask)
		return -ENOMEM;

	/*
	 * Use pi_lock to protect content of user_cpus_ptr
	 *
	 * Though unlikely, user_cpus_ptr can be reset to NULL by a concurrent
	 * do_set_cpus_allowed().
	 */
	raw_spin_lock_irqsave(&src->pi_lock, flags);
	if (src->user_cpus_ptr) {
		swap(dst->user_cpus_ptr, user_mask);
		cpumask_copy(dst->user_cpus_ptr, src->user_cpus_ptr);
	}
	raw_spin_unlock_irqrestore(&src->pi_lock, flags);

	if (unlikely(user_mask))
		kfree(user_mask);

	return 0;
}

/*
 * clear_user_cpus_ptr - 清空任务的用户亲和性指针，
 * 并返回原掩码供调用者处理。
 */
static inline struct cpumask *clear_user_cpus_ptr(struct task_struct *p)
{
	struct cpumask *user_mask = NULL;

	swap(p->user_cpus_ptr, user_mask);

	return user_mask;
}

/*
 * release_user_cpus_ptr - 清空并释放任务保存的用户亲和性掩码。
 */
void release_user_cpus_ptr(struct task_struct *p)
{
	kfree(clear_user_cpus_ptr(p));
}

/*
 * This function is wildly self concurrent; here be dragons.
 *
 *
 * When given a valid mask, __set_cpus_allowed_ptr() must block until the
 * designated task is enqueued on an allowed CPU. If that task is currently
 * running, we have to kick it out using the CPU stopper.
 *
 * Migrate-Disable comes along and tramples all over our nice sandcastle.
 * Consider:
 *
 *     Initial conditions: P0->cpus_mask = [0, 1]
 *
 *     P0@CPU0                  P1
 *
 *     migrate_disable();
 *     <preempted>
 *                              set_cpus_allowed_ptr(P0, [1]);
 *
 * P1 *cannot* return from this set_cpus_allowed_ptr() call until P0 executes
 * its outermost migrate_enable() (i.e. it exits its Migrate-Disable region).
 * This means we need the following scheme:
 *
 *     P0@CPU0                  P1
 *
 *     migrate_disable();
 *     <preempted>
 *                              set_cpus_allowed_ptr(P0, [1]);
 *                                <blocks>
 *     <resumes>
 *     migrate_enable();
 *       __set_cpus_allowed_ptr();
 *       <wakes local stopper>
 *                         `--> <woken on migration completion>
 *
 * Now the fun stuff: there may be several P1-like tasks, i.e. multiple
 * concurrent set_cpus_allowed_ptr(P0, [*]) calls. CPU affinity changes of any
 * task p are serialized by p->pi_lock, which we can leverage: the one that
 * should come into effect at the end of the Migrate-Disable region is the last
 * one. This means we only need to track a single cpumask (i.e. p->cpus_mask),
 * but we still need to properly signal those waiting tasks at the appropriate
 * moment.
 *
 * This is implemented using struct set_affinity_pending. The first
 * __set_cpus_allowed_ptr() caller within a given Migrate-Disable region will
 * setup an instance of that struct and install it on the targeted task_struct.
 * Any and all further callers will reuse that instance. Those then wait for
 * a completion signaled at the tail of the CPU stopper callback (1), triggered
 * on the end of the Migrate-Disable region (i.e. outermost migrate_enable()).
 *
 *
 * (1) In the cases covered above. There is one more where the completion is
 * signaled within affine_move_task() itself: when a subsequent affinity request
 * occurs after the stopper bailed out due to the targeted task still being
 * Migrate-Disable. Consider:
 *
 *     Initial conditions: P0->cpus_mask = [0, 1]
 *
 *     CPU0		  P1				P2
 *     <P0>
 *       migrate_disable();
 *       <preempted>
 *                        set_cpus_allowed_ptr(P0, [1]);
 *                          <blocks>
 *     <migration/0>
 *       migration_cpu_stop()
 *         is_migration_disabled()
 *           <bails>
 *                                                       set_cpus_allowed_ptr(P0, [0, 1]);
 *                                                         <signal completion>
 *                          <awakes>
 *
 * Note that the above is safe vs a concurrent migrate_enable(), as any
 * pending affinity completion is preceded by an uninstallation of
 * p->migration_pending done with p->pi_lock held.
 */
/*
 * affine_move_task - 落实亲和性变更，
 * 协调直接迁移、stopper 迁移及禁止迁移期间的等待。
 * 接管并释放 pi_lock 与运行队列锁，必要时等待共享迁移请求完成。
 */
static int affine_move_task(struct rq *rq, struct task_struct *p, struct rq_flags *rf,
			    int dest_cpu, unsigned int flags)
	__releases(rq->lock)
	__releases(p->pi_lock)
{
	struct set_affinity_pending my_pending = { }, *pending = NULL;
	bool stop_pending, complete = false;

	/*
	 * 三种后续动作：
	 *   已处于允许 CPU：通常直接结束，并处理此前等待者；
	 *   正执行或 TASK_WAKING：交给 stopper，避免与切换/异步入队竞态；
	 *   已排队且不执行：可直接搬队列，但 migration_disabled 时必须等。
	 * 普通调用者在锁外等待 done；migrate_enable 路径只提交工作，
	 * 让当前 CPU 的 stopper 得到执行机会，不能等待自身恢复迁移能力。
	 */
	/*
	 * Can the task run on the task's current CPU? If so, we're done
	 *
	 * We are also done if the task is the current donor, boosting a lock-
	 * holding proxy, (and potentially has been migrated outside its
	 * current or previous affinity mask)
	 */
	if (cpumask_test_cpu(task_cpu(p), &p->cpus_mask) ||
	    (task_current_donor(rq, p) && !task_current(rq, p))) {
		struct task_struct *push_task = NULL;

		if ((flags & SCA_MIGRATE_ENABLE) &&
		    (p->migration_flags & MDF_PUSH) && !rq->push_busy) {
			rq->push_busy = true;
			push_task = get_task_struct(p);
		}

		/*
		 * If there are pending waiters, but no pending stop_work,
		 * then complete now.
		 */
		pending = p->migration_pending;
		if (pending && !pending->stop_pending) {
			p->migration_pending = NULL;
			complete = true;
		}

		preempt_disable();
		task_rq_unlock(rq, p, rf);
		if (push_task) {
			stop_one_cpu_nowait(rq->cpu, push_cpu_stop,
					    p, &rq->push_work);
		}
		preempt_enable();

		if (complete)
			complete_all(&pending->done);

		return 0;
	}

	if (!(flags & SCA_MIGRATE_ENABLE)) {
		/* serialized by p->pi_lock */
		if (!p->migration_pending) {
			/*
			 * Install the request
			 *
			 * 请求在首个调用者栈上，其他调用者共享并增加 refs；
			 * 首个调用者必须等这些引用全部归还后才能返回。
			 */
			refcount_set(&my_pending.refs, 1);
			init_completion(&my_pending.done);
			my_pending.arg = (struct migration_arg) {
				.task = p,
				.dest_cpu = dest_cpu,
				.pending = &my_pending,
			};

			p->migration_pending = &my_pending;
		} else {
			pending = p->migration_pending;
			refcount_inc(&pending->refs);
			/*
			 * Affinity has changed, but we've already installed a
			 * pending. migration_cpu_stop() *must* see this, else
			 * we risk a completion of the pending despite having a
			 * task on a disallowed CPU.
			 *
			 * Serialized by p->pi_lock, so this is safe.
			 */
			pending->arg.dest_cpu = dest_cpu;
		}
	}
	pending = p->migration_pending;
	/*
	 * - !MIGRATE_ENABLE:
	 *   we'll have installed a pending if there wasn't one already.
	 *
	 * - MIGRATE_ENABLE:
	 *   we're here because the current CPU isn't matching anymore,
	 *   the only way that can happen is because of a concurrent
	 *   set_cpus_allowed_ptr() call, which should then still be
	 *   pending completion.
	 *
	 * Either way, we really should have a @pending here.
	 */
	if (WARN_ON_ONCE(!pending)) {
		task_rq_unlock(rq, p, rf);
		return -EINVAL;
	}

	if (task_on_cpu(rq, p) || READ_ONCE(p->__state) == TASK_WAKING) {
		/*
		 * MIGRATE_ENABLE gets here because 'p == current', but for
		 * anything else we cannot do is_migration_disabled(), punt
		 * and have the stopper function handle it all race-free.
		 */
		stop_pending = pending->stop_pending;
		if (!stop_pending)
			pending->stop_pending = true;

		if (flags & SCA_MIGRATE_ENABLE)
			p->migration_flags &= ~MDF_PUSH;

		preempt_disable();
		task_rq_unlock(rq, p, rf);
		if (!stop_pending) {
			stop_one_cpu_nowait(cpu_of(rq), migration_cpu_stop,
					    &pending->arg, &pending->stop_work);
		}
		preempt_enable();

		if (flags & SCA_MIGRATE_ENABLE)
			return 0;
	} else {

		if (!is_migration_disabled(p)) {
			if (task_on_rq_queued(p))
				rq = move_queued_task(rq, rf, p, dest_cpu);

			if (!pending->stop_pending) {
				p->migration_pending = NULL;
				complete = true;
			}
		}
		task_rq_unlock(rq, p, rf);

		if (complete)
			complete_all(&pending->done);
	}

	wait_for_completion(&pending->done);

	if (refcount_dec_and_test(&pending->refs))
		wake_up_var(&pending->refs); /* No UaF, just an address */

	/*
	 * Block the original owner of &pending until all subsequent callers
	 * have seen the completion and decremented the refcount
	 *
	 * done 只说明迁移条件已满足，不代表所有等待者都已停止访问请求。
	 * 首个调用者在这里守住栈对象的生命周期；其他调用者的
	 * my_pending 未被安装，refs 为零，不需要额外等待。
	 */
	wait_var_event(&my_pending.refs, !refcount_read(&my_pending.refs));

	/* ARGH */
	WARN_ON_ONCE(my_pending.stop_pending);

	return 0;
}

/*
 * Called with both p->pi_lock and rq->lock held; drops both before returning.
 */
/*
 * __set_cpus_allowed_ptr_locked - 在持锁状态下校验新亲和性，
 * 选取目标 CPU 并发起迁移。
 * 返回前释放任务和运行队列锁；失败返回负错误码。
 */
static int __set_cpus_allowed_ptr_locked(struct task_struct *p,
					 struct affinity_context *ctx,
					 struct rq *rq,
					 struct rq_flags *rf)
	__releases(rq->lock)
	__releases(p->pi_lock)
{
	const struct cpumask *cpu_allowed_mask = task_cpu_possible_mask(p);
	const struct cpumask *cpu_valid_mask = cpu_active_mask;
	bool kthread = p->flags & PF_KTHREAD;
	unsigned int dest_cpu;
	int ret = 0;

	update_rq_clock(rq);

	if (kthread || is_migration_disabled(p)) {
		/*
		 * Kernel threads are allowed on online && !active CPUs,
		 * however, during cpu-hot-unplug, even these might get pushed
		 * away if not KTHREAD_IS_PER_CPU.
		 *
		 * Specifically, migration_disabled() tasks must not fail the
		 * cpumask_any_and_distribute() pick below, esp. so on
		 * SCA_MIGRATE_ENABLE, otherwise we'll not call
		 * set_cpus_allowed_common() and actually reset p->cpus_ptr.
		 */
		cpu_valid_mask = cpu_online_mask;
	}

	if (!kthread && !cpumask_subset(ctx->new_mask, cpu_allowed_mask)) {
		ret = -EINVAL;
		goto out;
	}

	/*
	 * Must re-check here, to close a race against __kthread_bind(),
	 * sched_setaffinity() is not guaranteed to observe the flag.
	 */
	if ((ctx->flags & SCA_CHECK) && (p->flags & PF_NO_SETAFFINITY)) {
		ret = -EINVAL;
		goto out;
	}

	if (!(ctx->flags & SCA_MIGRATE_ENABLE)) {
		if (cpumask_equal(&p->cpus_mask, ctx->new_mask)) {
			if (ctx->flags & SCA_USER)
				swap(p->user_cpus_ptr, ctx->user_mask);
			goto out;
		}

		if (WARN_ON_ONCE(p == current &&
				 is_migration_disabled(p) &&
				 !cpumask_test_cpu(task_cpu(p), ctx->new_mask))) {
			ret = -EBUSY;
			goto out;
		}
	}

	/*
	 * Picking a ~random cpu helps in cases where we are changing affinity
	 * for groups of tasks (ie. cpuset), so that load balancing is not
	 * immediately required to distribute the tasks within their new mask.
	 */
	dest_cpu = cpumask_any_and_distribute(cpu_valid_mask, ctx->new_mask);
	if (dest_cpu >= nr_cpu_ids) {
		ret = -EINVAL;
		goto out;
	}

	/* 先重建亲和性约束，再把任务移动到满足约束的位置。 */
	__do_set_cpus_allowed(p, ctx);

	return affine_move_task(rq, p, rf, dest_cpu, ctx->flags);

out:
	task_rq_unlock(rq, p, rf);

	return ret;
}

/*
 * Change a given task's CPU affinity. Migrate the thread to a
 * proper CPU and schedule it away if the CPU it's executing on
 * is removed from the allowed bitmask.
 *
 * NOTE: the caller must have a valid reference to the task, the
 * task must not exit() & deallocate itself prematurely. The
 * call is not atomic; no spinlocks may be held.
 */
/*
 * __set_cpus_allowed_ptr - 加锁并结合用户掩码设置任务亲和性，
 * 必要时等待任务迁移。
 * 调用者须持有有效任务引用，且不得持有自旋锁。
 */
int __set_cpus_allowed_ptr(struct task_struct *p, struct affinity_context *ctx)
{
	struct rq_flags rf;
	struct rq *rq;

	rq = task_rq_lock(p, &rf);
	/*
	 * Masking should be skipped if SCA_USER or any of the SCA_MIGRATE_*
	 * flags are set.
	 */
	if (p->user_cpus_ptr &&
	    !(ctx->flags & (SCA_USER | SCA_MIGRATE_ENABLE | SCA_MIGRATE_DISABLE)) &&
	    cpumask_and(rq->scratch_mask, ctx->new_mask, p->user_cpus_ptr))
		ctx->new_mask = rq->scratch_mask;

	return __set_cpus_allowed_ptr_locked(p, ctx, rq, &rf);
}

/*
 * set_cpus_allowed_ptr - 以普通亲和性请求设置任务允许的 CPU 集合，
 * 返回操作结果。
 */
int set_cpus_allowed_ptr(struct task_struct *p, const struct cpumask *new_mask)
{
	struct affinity_context ac = {
		.new_mask  = new_mask,
		.flags     = 0,
	};

	return __set_cpus_allowed_ptr(p, &ac);
}
EXPORT_SYMBOL_GPL(set_cpus_allowed_ptr);

/*
 * Change a given task's CPU affinity to the intersection of its current
 * affinity mask and @subset_mask, writing the resulting mask to @new_mask.
 * If user_cpus_ptr is defined, use it as the basis for restricting CPU
 * affinity or use cpu_online_mask instead.
 *
 * If the resulting mask is empty, leave the affinity unchanged and return
 * -EINVAL.
 */
/*
 * restrict_cpus_allowed_ptr - 将任务亲和性收缩到给定集合，
 * 交集为空或不允许收缩时失败。
 */
static int restrict_cpus_allowed_ptr(struct task_struct *p,
				     struct cpumask *new_mask,
				     const struct cpumask *subset_mask)
{
	struct affinity_context ac = {
		.new_mask  = new_mask,
		.flags     = 0,
	};
	struct rq_flags rf;
	struct rq *rq;
	int err;

	rq = task_rq_lock(p, &rf);

	/*
	 * Forcefully restricting the affinity of a deadline task is
	 * likely to cause problems, so fail and noisily override the
	 * mask entirely.
	 */
	if (task_has_dl_policy(p) && dl_bandwidth_enabled()) {
		err = -EPERM;
		goto err_unlock;
	}

	if (!cpumask_and(new_mask, task_user_cpus(p), subset_mask)) {
		err = -EINVAL;
		goto err_unlock;
	}

	return __set_cpus_allowed_ptr_locked(p, &ac, rq, &rf);

err_unlock:
	task_rq_unlock(rq, p, &rf);
	return err;
}

/*
 * Restrict the CPU affinity of task @p so that it is a subset of
 * task_cpu_possible_mask() and point @p->user_cpus_ptr to a copy of the
 * old affinity mask. If the resulting mask is empty, we warn and walk
 * up the cpuset hierarchy until we find a suitable mask.
 */
/*
 * force_compatible_cpus_allowed_ptr - 强制任务使用兼容 CPU 集合，
 * 必要时按 cpuset 回退。
 * 保留用户亲和性请求，以便后续尝试恢复。
 */
void force_compatible_cpus_allowed_ptr(struct task_struct *p)
{
	cpumask_var_t new_mask;
	const struct cpumask *override_mask = task_cpu_possible_mask(p);

	alloc_cpumask_var(&new_mask, GFP_KERNEL);

	/*
	 * __migrate_task() can fail silently in the face of concurrent
	 * offlining of the chosen destination CPU, so take the hotplug
	 * lock to ensure that the migration succeeds.
	 */
	cpus_read_lock();
	if (!cpumask_available(new_mask))
		goto out_set_mask;

	if (!restrict_cpus_allowed_ptr(p, new_mask, override_mask))
		goto out_free_mask;

	/*
	 * We failed to find a valid subset of the affinity mask for the
	 * task, so override it based on its cpuset hierarchy.
	 */
	cpuset_cpus_allowed(p, new_mask);
	override_mask = new_mask;

out_set_mask:
	if (printk_ratelimit()) {
		printk_deferred("Overriding affinity for process %d (%s) to CPUs %*pbl\n",
				task_pid_nr(p), p->comm,
				cpumask_pr_args(override_mask));
	}

	WARN_ON(set_cpus_allowed_ptr(p, override_mask));
out_free_mask:
	cpus_read_unlock();
	free_cpumask_var(new_mask);
}

/*
 * Restore the affinity of a task @p which was previously restricted by a
 * call to force_compatible_cpus_allowed_ptr().
 *
 * It is the caller's responsibility to serialise this with any calls to
 * force_compatible_cpus_allowed_ptr(@p).
 */
/*
 * relax_compatible_cpus_allowed_ptr - 尝试恢复此前因 CPU 兼容性
 * 而受限的用户亲和性。
 * 调用者须与强制收缩亲和性的操作串行化。
 */
void relax_compatible_cpus_allowed_ptr(struct task_struct *p)
{
	struct affinity_context ac = {
		.new_mask  = task_user_cpus(p),
		.flags     = 0,
	};
	int ret;

	/*
	 * Try to restore the old affinity mask with __sched_setaffinity().
	 * Cpuset masking will be done there too.
	 */
	ret = __sched_setaffinity(p, &ac);
	WARN_ON_ONCE(ret);
}

#ifdef CONFIG_SMP

/*
 * set_task_cpu - 更新任务所属 CPU，
 * 并通知调度类、rseq、性能事件及并发 ID 迁移逻辑。
 * 仅用于可运行或唤醒中的任务，调用者须提供 pi_lock 或运行队列锁保护。
 */
void set_task_cpu(struct task_struct *p, unsigned int new_cpu)
{
	unsigned int state = READ_ONCE(p->__state);

	/*
	 * We should never call set_task_cpu() on a blocked task,
	 * ttwu() will sort out the placement.
	 */
	WARN_ON_ONCE(state != TASK_RUNNING && state != TASK_WAKING && !p->on_rq);

	/*
	 * Migrating fair class task must have p->on_rq = TASK_ON_RQ_MIGRATING,
	 * because schedstat_wait_{start,end} rebase migrating task's wait_start
	 * time relying on p->on_rq.
	 */
	WARN_ON_ONCE(state == TASK_RUNNING &&
		     p->sched_class == &fair_sched_class &&
		     (p->on_rq && !task_on_rq_migrating(p)));

#ifdef CONFIG_LOCKDEP
	/*
	 * The caller should hold either p->pi_lock or rq->lock, when changing
	 * a task's CPU. ->pi_lock for waking tasks, rq->lock for runnable tasks.
	 *
	 * sched_move_task() holds both and thus holding either pins the cgroup,
	 * see task_group().
	 *
	 * Furthermore, all task_rq users should acquire both locks, see
	 * task_rq_lock().
	 */
	WARN_ON_ONCE(debug_locks && !(lockdep_is_held(&p->pi_lock) ||
				      lockdep_is_held(__rq_lockp(task_rq(p)))));
#endif
	/*
	 * Clearly, migrating tasks to offline CPUs is a fairly daft thing.
	 */
	WARN_ON_ONCE(!cpu_online(new_cpu));

	WARN_ON_ONCE(is_migration_disabled(p));

	trace_sched_migrate_task(p, new_cpu);

	if (task_cpu(p) != new_cpu) {
		/*
		 * 类回调在 CPU 字段改变前运行，便于从源 rq 转换时钟/负载。
		 * rseq 要在返回用户态前中止依赖旧 CPU 的临界区；MM CID 只在
		 * 此处记录源 CPU，目标队列激活时才完成编号转移或回收。
		 */
		if (p->sched_class->migrate_task_rq)
			p->sched_class->migrate_task_rq(p, new_cpu);
		p->se.nr_migrations++;
		rseq_migrate(p);
		sched_mm_cid_migrate_from(p);
		perf_event_task_migrate(p);
	}

	__set_task_cpu(p, new_cpu);
}
#endif /* CONFIG_SMP */

#ifdef CONFIG_NUMA_BALANCING
/*
 * __migrate_swap_task - 在交换迁移中移动已排队任务，
 * 睡眠任务则只更新下次唤醒 CPU。
 */
static void __migrate_swap_task(struct task_struct *p, int cpu)
{
	if (task_on_rq_queued(p)) {
		struct rq *src_rq, *dst_rq;
		struct rq_flags srf, drf;

		src_rq = task_rq(p);
		dst_rq = cpu_rq(cpu);

		rq_pin_lock(src_rq, &srf);
		rq_pin_lock(dst_rq, &drf);

		move_queued_task_locked(src_rq, dst_rq, p);
		wakeup_preempt(dst_rq, p, 0);

		rq_unpin_lock(dst_rq, &drf);
		rq_unpin_lock(src_rq, &srf);

	} else {
		/*
		 * Task isn't running anymore; make it appear like we migrated
		 * it before it went to sleep. This means on wakeup we make the
		 * previous CPU our target instead of where it really is.
		 */
		p->wake_cpu = cpu;
	}
}

struct migration_swap_arg {
	struct task_struct *src_task, *dst_task;
	int src_cpu, dst_cpu;
};

/*
 * migrate_swap_stop - 锁定两个任务及队列，验证 CPU 和亲和性后交换任务位置。
 * 状态已变化时返回 -EAGAIN，供上层重试。
 */
static int migrate_swap_stop(void *data)
{
	struct migration_swap_arg *arg = data;
	struct rq *src_rq, *dst_rq;

	if (!cpu_active(arg->src_cpu) || !cpu_active(arg->dst_cpu))
		return -EAGAIN;

	src_rq = cpu_rq(arg->src_cpu);
	dst_rq = cpu_rq(arg->dst_cpu);

	guard(double_raw_spinlock)(&arg->src_task->pi_lock, &arg->dst_task->pi_lock);
	guard(double_rq_lock)(src_rq, dst_rq);

	if (task_cpu(arg->dst_task) != arg->dst_cpu)
		return -EAGAIN;

	if (task_cpu(arg->src_task) != arg->src_cpu)
		return -EAGAIN;

	if (!cpumask_test_cpu(arg->dst_cpu, arg->src_task->cpus_ptr))
		return -EAGAIN;

	if (!cpumask_test_cpu(arg->src_cpu, arg->dst_task->cpus_ptr))
		return -EAGAIN;

	__migrate_swap_task(arg->src_task, arg->dst_cpu);
	__migrate_swap_task(arg->dst_task, arg->src_cpu);

	return 0;
}

/*
 * Cross migrate two tasks
 */
/*
 * migrate_swap - 使用双 CPU stopper 交换两个任务所在的 CPU，服务于 NUMA 均衡。
 */
int migrate_swap(struct task_struct *cur, struct task_struct *p,
		int target_cpu, int curr_cpu)
{
	struct migration_swap_arg arg;
	int ret = -EINVAL;

	arg = (struct migration_swap_arg){
		.src_task = cur,
		.src_cpu = curr_cpu,
		.dst_task = p,
		.dst_cpu = target_cpu,
	};

	if (arg.src_cpu == arg.dst_cpu)
		goto out;

	/*
	 * These three tests are all lockless; this is OK since all of them
	 * will be re-checked with proper locks held further down the line.
	 */
	if (!cpu_active(arg.src_cpu) || !cpu_active(arg.dst_cpu))
		goto out;

	if (!cpumask_test_cpu(arg.dst_cpu, arg.src_task->cpus_ptr))
		goto out;

	if (!cpumask_test_cpu(arg.src_cpu, arg.dst_task->cpus_ptr))
		goto out;

	trace_sched_swap_numa(cur, arg.src_cpu, p, arg.dst_cpu);
	ret = stop_two_cpus(arg.dst_cpu, arg.src_cpu, migrate_swap_stop, &arg);

out:
	return ret;
}
#endif /* CONFIG_NUMA_BALANCING */

/***
 * kick_process - kick a running thread to enter/exit the kernel
 * 向远端正在执行该任务的 CPU 发送重调度 IPI，
 * 使任务及时进入内核处理事件。
 * @p: the to-be-kicked thread
 *
 * Cause a process which is running on another CPU to enter
 * kernel-mode, without any delay. (to get signals handled.)
 *
 * NOTE: this function doesn't have to take the runqueue lock,
 * because all it wants to ensure is that the remote task enters
 * the kernel. If the IPI races and the task has been migrated
 * to another CPU then no harm is done and the purpose has been
 * achieved as well.
 */
void kick_process(struct task_struct *p)
{
	guard(preempt)();
	int cpu = task_cpu(p);

	if ((cpu != smp_processor_id()) && task_curr(p))
		smp_send_reschedule(cpu);
}
EXPORT_SYMBOL_GPL(kick_process);

/*
 * ->cpus_ptr is protected by both rq->lock and p->pi_lock
 *
 * A few notes on cpu_active vs cpu_online:
 *
 *  - cpu_active must be a subset of cpu_online
 *
 *  - on CPU-up we allow per-CPU kthreads on the online && !active CPU,
 *    see __set_cpus_allowed_ptr(). At this point the newly online
 *    CPU isn't yet part of the sched domains, and balancing will not
 *    see it.
 *
 *  - on CPU-down we clear cpu_active() to mask the sched domains and
 *    avoid the load balancer to place new tasks on the to be removed
 *    CPU. Existing tasks will remain running there and will be taken
 *    off.
 *
 * This means that fallback selection must not select !active CPUs.
 * And can assume that any active CPU must be online. Conversely
 * select_task_rq() below may allow selection of !active CPUs in order
 * to satisfy the above rules.
 */
/*
 * select_fallback_rq - 原 CPU 不可用时选择合法替代 CPU，优先同 NUMA 节点。
 * 无候选时逐步放宽 cpuset 和亲和性约束，保证任务可被调度。
 */
static int select_fallback_rq(int cpu, struct task_struct *p)
{
	int nid = cpu_to_node(cpu);
	const struct cpumask *nodemask = NULL;
	enum { cpuset, possible, fail } state = cpuset;
	int dest_cpu;

	/*
	 * 优先同 NUMA 节点，再扫描现有允许集合；都失败才依次放宽
	 * cpuset 和 CPU 亲和性。这里是保证任务能继续运行的兜底路径，
	 * 并不是调度类用于优化负载、容量或缓存局部性的常规选核算法。
	 */
	/*
	 * If the node that the CPU is on has been offlined, cpu_to_node()
	 * will return -1. There is no CPU on the node, and we should
	 * select the CPU on the other node.
	 */
	if (nid != -1) {
		nodemask = cpumask_of_node(nid);

		/* Look for allowed, online CPU in same node. */
		for_each_cpu(dest_cpu, nodemask) {
			if (is_cpu_allowed(p, dest_cpu))
				return dest_cpu;
		}
	}

	for (;;) {
		/* Any allowed, online CPU? */
		for_each_cpu(dest_cpu, p->cpus_ptr) {
			if (!is_cpu_allowed(p, dest_cpu))
				continue;

			goto out;
		}

		/* No more Mr. Nice Guy. */
		switch (state) {
		case cpuset:
			if (cpuset_cpus_allowed_fallback(p)) {
				state = possible;
				break;
			}
			fallthrough;
		case possible:
			/*
			 * XXX When called from select_task_rq() we only
			 * hold p->pi_lock and again violate locking order.
			 *
			 * More yuck to audit.
			 */
			do_set_cpus_allowed(p, task_cpu_fallback_mask(p));
			state = fail;
			break;
		case fail:
			BUG();
			break;
		}
	}

out:
	if (state != cpuset) {
		/*
		 * Don't tell them about moving exiting tasks or
		 * kernel threads (both mm NULL), since they never
		 * leave kernel.
		 */
		if (p->mm && printk_ratelimit()) {
			printk_deferred("process %d (%s) no longer affine to cpu%d\n",
					task_pid_nr(p), p->comm, cpu);
		}
	}

	return dest_cpu;
}

/*
 * The caller (fork, wakeup) owns p->pi_lock, ->cpus_ptr is stable.
 */
/*
 * select_task_rq - 让调度类为 fork 或唤醒选取 CPU，并以通用规则校验和回退。
 * 调用者须持有 pi_lock；wake_flags 会记录是否执行了调度类选核。
 */
static inline
int select_task_rq(struct task_struct *p, int cpu, int *wake_flags)
{
	lockdep_assert_held(&p->pi_lock);

	if (p->nr_cpus_allowed > 1 && !is_migration_disabled(p)) {
		cpu = p->sched_class->select_task_rq(p, cpu, *wake_flags);
		*wake_flags |= WF_RQ_SELECTED;
	} else {
		cpu = cpumask_any(p->cpus_ptr);
	}

	/*
	 * In order not to call set_task_cpu() on a blocking task we need
	 * to rely on ttwu() to place the task on a valid ->cpus_ptr
	 * CPU.
	 *
	 * Since this is common to all placement strategies, this lives here.
	 *
	 * [ this allows ->select_task() to simply return task_cpu(p) and
	 *   not worry about this generic constraint ]
	 */
	if (unlikely(!is_cpu_allowed(p, cpu)))
		cpu = select_fallback_rq(task_cpu(p), p);

	return cpu;
}

/*
 * sched_set_stop_task - 安装指定 CPU 的 stopper 任务，
 * 设置专用调度类及锁依赖类型。
 * 被替换的旧 stopper 恢复实时调度类，以便正常退出。
 */
void sched_set_stop_task(int cpu, struct task_struct *stop)
{
	static struct lock_class_key stop_pi_lock;
	struct sched_param param = { .sched_priority = MAX_RT_PRIO - 1 };
	struct task_struct *old_stop = cpu_rq(cpu)->stop;

	if (stop) {
		/*
		 * Make it appear like a SCHED_FIFO task, its something
		 * userspace knows about and won't get confused about.
		 *
		 * Also, it will make PI more or less work without too
		 * much confusion -- but then, stop work should not
		 * rely on PI working anyway.
		 */
		sched_setscheduler_nocheck(stop, SCHED_FIFO, &param);

		stop->sched_class = &stop_sched_class;

		/*
		 * The PI code calls rt_mutex_setprio() with ->pi_lock held to
		 * adjust the effective priority of a task. As a result,
		 * rt_mutex_setprio() can trigger (RT) balancing operations,
		 * which can then trigger wakeups of the stop thread to push
		 * around the current task.
		 *
		 * The stop task itself will never be part of the PI-chain, it
		 * never blocks, therefore that ->pi_lock recursion is safe.
		 * Tell lockdep about this by placing the stop->pi_lock in its
		 * own class.
		 */
		lockdep_set_class(&stop->pi_lock, &stop_pi_lock);
	}

	cpu_rq(cpu)->stop = stop;

	if (old_stop) {
		/*
		 * Reset it back to a normal scheduling class so that
		 * it can die in pieces.
		 */
		old_stop->sched_class = &rt_sched_class;
	}
}

/*
 * ttwu_stat - 记录本地、远端、迁移及同步唤醒的任务和调度域统计。
 */
static void
ttwu_stat(struct task_struct *p, int cpu, int wake_flags)
{
	struct rq *rq;

	if (!schedstat_enabled())
		return;

	rq = this_rq();

	if (cpu == rq->cpu) {
		__schedstat_inc(rq->ttwu_local);
		__schedstat_inc(p->stats.nr_wakeups_local);
	} else {
		struct sched_domain *sd;

		__schedstat_inc(p->stats.nr_wakeups_remote);

		guard(rcu)();
		for_each_domain(rq->cpu, sd) {
			if (cpumask_test_cpu(cpu, sched_domain_span(sd))) {
				__schedstat_inc(sd->ttwu_wake_remote);
				break;
			}
		}
	}

	if (wake_flags & WF_MIGRATED)
		__schedstat_inc(p->stats.nr_wakeups_migrate);

	__schedstat_inc(rq->ttwu_count);
	__schedstat_inc(p->stats.nr_wakeups);

	if (wake_flags & WF_SYNC)
		__schedstat_inc(p->stats.nr_wakeups_sync);
}

/*
 * Mark the task runnable.
 */
/*
 * ttwu_do_wakeup - 将任务状态发布为 TASK_RUNNING，并发出任务唤醒跟踪事件。
 */
static inline void ttwu_do_wakeup(struct task_struct *p)
{
	WRITE_ONCE(p->__state, TASK_RUNNING);
	trace_sched_wakeup(p);
}

/*
 * ttwu_do_activate - 完成唤醒入队、等待记账、抢占检查及调度类唤醒回调。
 * 调用者须持有运行队列锁。
 */
static void
ttwu_do_activate(struct rq *rq, struct task_struct *p, int wake_flags,
		 struct rq_flags *rf)
{
	int en_flags = ENQUEUE_WAKEUP | ENQUEUE_NOCLOCK;

	lockdep_assert_rq_held(rq);

	/*
	 * nr_uninterruptible 只保证跨 CPU 求和有意义：睡眠时可能在源 rq
	 * 增加、唤醒时在目标 rq 减少，不能把单个 rq 的值当本地睡眠者数。
	 */
	if (p->sched_contributes_to_load)
		rq->nr_uninterruptible--;

	if (wake_flags & WF_RQ_SELECTED)
		en_flags |= ENQUEUE_RQ_SELECTED;
	if (wake_flags & WF_MIGRATED)
		en_flags |= ENQUEUE_MIGRATED;
	else
	if (p->in_iowait) {
		delayacct_blkio_end(p);
		atomic_dec(&task_rq(p)->nr_iowait);
	}

	/*
	 * 在 rq 锁内先完成入队和抢占检查，再发布 TASK_RUNNING。
	 * TASK_RUNNING 表示可运行，并不表示 p 已成为 rq->curr。
	 */
	activate_task(rq, p, en_flags);
	wakeup_preempt(rq, p, wake_flags);

	ttwu_do_wakeup(p);

	if (p->sched_class->task_woken) {
		/*
		 * Our task @p is fully woken up and running; so it's safe to
		 * drop the rq->lock, hereafter rq is only used for statistics.
		 */
		rq_unpin_lock(rq, rf);
		p->sched_class->task_woken(rq, p);
		rq_repin_lock(rq, rf);
	}

	if (rq->idle_stamp) {
		u64 delta = rq_clock(rq) - rq->idle_stamp;
		u64 max = 2*rq->max_idle_balance_cost;

		update_avg(&rq->avg_idle, delta);

		if (rq->avg_idle > max)
			rq->avg_idle = max;

		rq->idle_stamp = 0;
	}
}

/*
 * Consider @p being inside a wait loop:
 *
 *   for (;;) {
 *      set_current_state(TASK_UNINTERRUPTIBLE);
 *
 *      if (CONDITION)
 *         break;
 *
 *      schedule();
 *   }
 *   __set_current_state(TASK_RUNNING);
 *
 * between set_current_state() and schedule(). In this case @p is still
 * runnable, so all that needs doing is change p->state back to TASK_RUNNING in
 * an atomic manner.
 *
 * By taking task_rq(p)->lock we serialize against schedule(), if @p->on_rq
 * then schedule() must still happen and p->state can be changed to
 * TASK_RUNNING. Otherwise we lost the race, schedule() has happened, and we
 * need to do a full wakeup with enqueue.
 *
 * Returns: %true when the wakeup is done,
 *          %false otherwise.
 */
/*
 * ttwu_runnable - 对尚未离开运行队列的任务完成快速唤醒，避免重复入队。
 * 返回 1 表示唤醒完成，返回 0 表示需走完整激活路径。
 */
static int ttwu_runnable(struct task_struct *p, int wake_flags)
{
	struct rq_flags rf;
	struct rq *rq;
	int ret = 0;

	rq = __task_rq_lock(p, &rf);
	if (task_on_rq_queued(p)) {
		update_rq_clock(rq);
		/*
		 * 延迟出队的实体仍在公平队列中，但此前已不被视为可运行。
		 * 用 ENQUEUE_DELAYED 恢复它，而不是按全新实体重复插入。
		 */
		if (p->se.sched_delayed)
			enqueue_task(rq, p, ENQUEUE_NOCLOCK | ENQUEUE_DELAYED);
		if (!task_on_cpu(rq, p)) {
			/*
			 * When on_rq && !on_cpu the task is preempted, see if
			 * it should preempt the task that is current now.
			 */
			wakeup_preempt(rq, p, wake_flags);
		}
		ttwu_do_wakeup(p);
		ret = 1;
	}
	__task_rq_unlock(rq, &rf);

	return ret;
}

/*
 * sched_ttwu_pending - 在目标 CPU 批量激活远端排入唤醒链表的任务，
 * 并清除待唤醒标志。
 */
void sched_ttwu_pending(void *arg)
{
	struct llist_node *llist = arg;
	struct rq *rq = this_rq();
	struct task_struct *p, *t;
	struct rq_flags rf;

	if (!llist)
		return;

	/* 批次共用一次取锁和时钟更新；每个任务已由 TASK_WAKING 认领。 */
	rq_lock_irqsave(rq, &rf);
	update_rq_clock(rq);

	llist_for_each_entry_safe(p, t, llist, wake_entry.llist) {
		if (WARN_ON_ONCE(p->on_cpu))
			smp_cond_load_acquire(&p->on_cpu, !VAL);

		if (WARN_ON_ONCE(task_cpu(p) != cpu_of(rq)))
			set_task_cpu(p, cpu_of(rq));

		ttwu_do_activate(rq, p, p->sched_remote_wakeup ? WF_MIGRATED : 0, &rf);
	}

	/*
	 * Must be after enqueueing at least once task such that
	 * idle_cpu() does not observe a false-negative -- if it does,
	 * it is possible for select_idle_siblings() to stack a number
	 * of tasks on this CPU during that window.
	 *
	 * It is OK to clear ttwu_pending when another task pending.
	 * We will receive IPI after local IRQ enabled and then enqueue it.
	 * Since now nr_running > 0, idle_cpu() will always get correct result.
	 *
	 * ttwu_pending 填补了“已提交远端唤醒，但尚未增加 nr_running”
	 * 的窗口。先入队再清标记，才能避免其他 CPU 把这里误判为空闲，
	 * 连续把多个唤醒任务都堆到同一个 CPU。
	 */
	WRITE_ONCE(rq->ttwu_pending, 0);
	rq_unlock_irqrestore(rq, &rf);
}

/*
 * Prepare the scene for sending an IPI for a remote smp_call
 *
 * Returns true if the caller can proceed with sending the IPI.
 * Returns false otherwise.
 */
/*
 * call_function_single_prep_ipi - 为远端调用准备通知，
 * 轮询空闲任务可自行处理时省略 IPI。
 * 返回 true 表示调用者仍需发送 IPI。
 */
bool call_function_single_prep_ipi(int cpu)
{
	if (set_nr_if_polling(cpu_rq(cpu)->idle)) {
		trace_sched_wake_idle_without_ipi(cpu);
		return false;
	}

	return true;
}

/*
 * Queue a task on the target CPUs wake_list and wake the CPU via IPI if
 * necessary. The wakee CPU on receipt of the IPI will queue the task
 * via sched_ttwu_wakeup() for activation so the wakee incurs the cost
 * of the wakeup instead of the waker.
 */
/*
 * __ttwu_queue_wakelist - 将唤醒任务挂到目标 CPU 的异步调用链表，
 * 并记录迁移标志。
 */
static void __ttwu_queue_wakelist(struct task_struct *p, int cpu, int wake_flags)
{
	struct rq *rq = cpu_rq(cpu);

	p->sched_remote_wakeup = !!(wake_flags & WF_MIGRATED);

	WRITE_ONCE(rq->ttwu_pending, 1);
#ifdef CONFIG_SMP
	__smp_call_single_queue(cpu, &p->wake_entry.llist);
#endif
}

/*
 * wake_up_if_idle - 加锁复核目标 CPU 仍在执行空闲任务后，发出重调度请求。
 */
void wake_up_if_idle(int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	guard(rcu)();
	if (is_idle_task(rcu_dereference(rq->curr))) {
		guard(rq_lock_irqsave)(rq);
		if (is_idle_task(rq->curr))
			resched_curr(rq);
	}
}

/*
 * cpus_equal_capacity - 判断两个 CPU 的原始计算能力是否相同。
 */
bool cpus_equal_capacity(int this_cpu, int that_cpu)
{
	if (!sched_asym_cpucap_active())
		return true;

	if (this_cpu == that_cpu)
		return true;

	return arch_scale_cpu_capacity(this_cpu) == arch_scale_cpu_capacity(that_cpu);
}

/*
 * cpus_share_cache - 判断两个 CPU 是否属于同一末级缓存共享域。
 */
bool cpus_share_cache(int this_cpu, int that_cpu)
{
	if (this_cpu == that_cpu)
		return true;

	return per_cpu(sd_llc_id, this_cpu) == per_cpu(sd_llc_id, that_cpu);
}

/*
 * Whether CPUs are share cache resources, which means LLC on non-cluster
 * machines and LLC tag or L2 on machines with clusters.
 */
/*
 * cpus_share_resources - 判断两个 CPU 是否共享调度器定义的缓存资源域。
 */
bool cpus_share_resources(int this_cpu, int that_cpu)
{
	if (this_cpu == that_cpu)
		return true;

	return per_cpu(sd_share_id, this_cpu) == per_cpu(sd_share_id, that_cpu);
}

/*
 * ttwu_queue_cond - 综合调度类许可、CPU 状态、亲和性与缓存局部性，
 * 选择异步唤醒。
 */
static inline bool ttwu_queue_cond(struct task_struct *p, int cpu)
{
	/*
	 * 这是“谁来做入队工作”的选择，不是“任务去哪个 CPU”的选择；
	 * cpu 已由上层给出。跨 LLC 直接改远端 rq 会来回传递缓存行，
	 * 让目标 CPU 自己入队通常更划算，但热插拔和类策略有优先权。
	 */
	/* See SCX_OPS_ALLOW_QUEUED_WAKEUP. */
	if (!scx_allow_ttwu_queue(p))
		return false;

#ifdef CONFIG_SMP
	if (p->sched_class == &stop_sched_class)
		return false;
#endif

	/*
	 * Do not complicate things with the async wake_list while the CPU is
	 * in hotplug state.
	 */
	if (!cpu_active(cpu))
		return false;

	/* Ensure the task will still be allowed to run on the CPU. */
	if (!cpumask_test_cpu(cpu, p->cpus_ptr))
		return false;

	/*
	 * If the CPU does not share cache, then queue the task on the
	 * remote rqs wakelist to avoid accessing remote data.
	 */
	if (!cpus_share_cache(smp_processor_id(), cpu))
		return true;

	if (cpu == smp_processor_id())
		return false;

	/*
	 * If the wakee cpu is idle, or the task is descheduling and the
	 * only running task on the CPU, then use the wakelist to offload
	 * the task activation to the idle (or soon-to-be-idle) CPU as
	 * the current CPU is likely busy. nr_running is checked to
	 * avoid unnecessary task stacking.
	 *
	 * Note that we can only get here with (wakee) p->on_rq=0,
	 * p->on_cpu can be whatever, we've done the dequeue, so
	 * the wakee has been accounted out of ->nr_running.
	 */
	if (!cpu_rq(cpu)->nr_running)
		return true;

	return false;
}

/*
 * ttwu_queue_wakelist - 条件合适时同步时钟并提交异步唤醒，
 * 成功提交返回 true。
 */
static bool ttwu_queue_wakelist(struct task_struct *p, int cpu, int wake_flags)
{
	if (sched_feat(TTWU_QUEUE) && ttwu_queue_cond(p, cpu)) {
		sched_clock_cpu(cpu); /* Sync clocks across CPUs */
		__ttwu_queue_wakelist(p, cpu, wake_flags);
		return true;
	}

	return false;
}

/*
 * ttwu_queue - 优先把唤醒处理卸载给目标 CPU，否则直接加锁并激活任务。
 */
static void ttwu_queue(struct task_struct *p, int cpu, int wake_flags)
{
	struct rq *rq = cpu_rq(cpu);
	struct rq_flags rf;

	/*
	 * 返回前可能仅完成“提交”，任务仍是 TASK_WAKING；此时由目标
	 * CPU 负责最终入队。同步路径则由唤醒者跨 CPU 取得目标 rq 锁。
	 */
	if (ttwu_queue_wakelist(p, cpu, wake_flags))
		return;

	rq_lock(rq, &rf);
	update_rq_clock(rq);
	ttwu_do_activate(rq, p, wake_flags, &rf);
	rq_unlock(rq, &rf);
}

/*
 * Invoked from try_to_wake_up() to check whether the task can be woken up.
 *
 * The caller holds p::pi_lock if p != current or has preemption
 * disabled when p == current.
 *
 * The rules of saved_state:
 *
 *   The related locking code always holds p::pi_lock when updating
 *   p::saved_state, which means the code is fully serialized in both cases.
 *
 *   For PREEMPT_RT, the lock wait and lock wakeups happen via TASK_RTLOCK_WAIT.
 *   No other bits set. This allows to distinguish all wakeup scenarios.
 *
 *   For FREEZER, the wakeup happens via TASK_FROZEN. No other bits set. This
 *   allows us to prevent early wakeup of tasks before they can be run on
 *   asymmetric ISA architectures (eg ARMv9).
 */
/*
 * ttwu_state_match - 判断是否应实际唤醒任务，
 * 并通过 success 返回逻辑唤醒结果。
 * 仅保存状态匹配时更新 saved_state，避免丢失 RT 锁或冻结期间的普通唤醒。
 */
static __always_inline
bool ttwu_state_match(struct task_struct *p, unsigned int state, int *success)
{
	int match;

	if (IS_ENABLED(CONFIG_DEBUG_PREEMPT)) {
		WARN_ON_ONCE((state & TASK_RTLOCK_WAIT) &&
			     state != TASK_RTLOCK_WAIT);
	}

	*success = !!(match = __task_state_match(p, state));

	/*
	 * Saved state preserves the task state across blocking on
	 * an RT lock or TASK_FREEZABLE tasks.  If the state matches,
	 * set p::saved_state to TASK_RUNNING, but do not wake the task
	 * because it waits for a lock wakeup or __thaw_task(). Also
	 * indicate success because from the regular waker's point of
	 * view this has succeeded.
	 *
	 * After acquiring the lock the task will restore p::__state
	 * from p::saved_state which ensures that the regular
	 * wakeup is not lost. The restore will also set
	 * p::saved_state to TASK_RUNNING so any further tests will
	 * not result in false positives vs. @success
	 */
	if (match < 0)
		p->saved_state = TASK_RUNNING;

	return match > 0;
}

/*
 * Notes on Program-Order guarantees on SMP systems.
 *
 *  MIGRATION
 *
 * The basic program-order guarantee on SMP systems is that when a task [t]
 * migrates, all its activity on its old CPU [c0] happens-before any subsequent
 * execution on its new CPU [c1].
 *
 * For migration (of runnable tasks) this is provided by the following means:
 *
 *  A) UNLOCK of the rq(c0)->lock scheduling out task t
 *  B) migration for t is required to synchronize *both* rq(c0)->lock and
 *     rq(c1)->lock (if not at the same time, then in that order).
 *  C) LOCK of the rq(c1)->lock scheduling in task
 *
 * Release/acquire chaining guarantees that B happens after A and C after B.
 * Note: the CPU doing B need not be c0 or c1
 *
 * Example:
 *
 *   CPU0            CPU1            CPU2
 *
 *   LOCK rq(0)->lock
 *   sched-out X
 *   sched-in Y
 *   UNLOCK rq(0)->lock
 *
 *                                   LOCK rq(0)->lock // orders against CPU0
 *                                   dequeue X
 *                                   UNLOCK rq(0)->lock
 *
 *                                   LOCK rq(1)->lock
 *                                   enqueue X
 *                                   UNLOCK rq(1)->lock
 *
 *                   LOCK rq(1)->lock // orders against CPU2
 *                   sched-out Z
 *                   sched-in X
 *                   UNLOCK rq(1)->lock
 *
 *
 *  BLOCKING -- aka. SLEEP + WAKEUP
 *
 * For blocking we (obviously) need to provide the same guarantee as for
 * migration. However the means are completely different as there is no lock
 * chain to provide order. Instead we do:
 *
 *   1) smp_store_release(X->on_cpu, 0)   -- finish_task()
 *   2) smp_cond_load_acquire(!X->on_cpu) -- try_to_wake_up()
 *
 * Example:
 *
 *   CPU0 (schedule)  CPU1 (try_to_wake_up) CPU2 (schedule)
 *
 *   LOCK rq(0)->lock LOCK X->pi_lock
 *   dequeue X
 *   sched-out X
 *   smp_store_release(X->on_cpu, 0);
 *
 *                    smp_cond_load_acquire(&X->on_cpu, !VAL);
 *                    X->state = WAKING
 *                    set_task_cpu(X,2)
 *
 *                    LOCK rq(2)->lock
 *                    enqueue X
 *                    X->state = RUNNING
 *                    UNLOCK rq(2)->lock
 *
 *                                          LOCK rq(2)->lock // orders against CPU1
 *                                          sched-out Z
 *                                          sched-in X
 *                                          UNLOCK rq(2)->lock
 *
 *                    UNLOCK X->pi_lock
 *   UNLOCK rq(0)->lock
 *
 *
 * However, for wakeups there is a second guarantee we must provide, namely we
 * must ensure that CONDITION=1 done by the caller can not be reordered with
 * accesses to the task state; see try_to_wake_up() and set_current_state().
 */

/**
 * try_to_wake_up - wake up a thread
 * 按状态掩码唤醒任务，协调并发唤醒、迁移和入队，
 * 成功唤醒时返回非零。
 * @p: the thread to be awakened
 * @state: the mask of task states that can be woken
 * @wake_flags: wake modifier flags (WF_*)
 *
 * Conceptually does:
 *
 *   If (@state & @p->state) @p->state = TASK_RUNNING.
 *
 * If the task was not queued/runnable, also place it back on a runqueue.
 *
 * This function is atomic against schedule() which would dequeue the task.
 *
 * It issues a full memory barrier before accessing @p->state, see the comment
 * with set_current_state().
 *
 * Uses p->pi_lock to serialize against concurrent wake-ups.
 *
 * Relies on p->pi_lock stabilizing:
 *  - p->sched_class
 *  - p->cpus_ptr
 *  - p->sched_task_group
 * in order to do migration, see its use of select_task_rq()/set_task_cpu().
 *
 * Tries really hard to only take one task_rq(p)->lock for performance.
 * Takes rq->lock in:
 *  - ttwu_runnable()    -- old rq, unavoidable, see comment there;
 *  - ttwu_queue()       -- new rq, for enqueue of the task;
 *  - psi_ttwu_dequeue() -- much sadness :-( accounting will kill us.
 *
 * As a consequence we race really badly with just about everything. See the
 * many memory barriers and their comments for details.
 *
 * Return: %true if @p->state changes (an actual wakeup was done),
 *	   %false otherwise.
 */
/*
 * 普通睡眠/唤醒的时间轴（延迟出队和代理执行另见对应分支）：
 *   等待方：设置睡眠 __state -> 检查条件 -> schedule -> on_rq=0
 *           -> 切到另一任务 -> finish_task 发布 on_cpu=0。
 *   唤醒方：发布条件 -> 匹配 __state -> 检查 on_rq
 *           -> 已在队列则恢复状态，否则认领为 TASK_WAKING
 *           -> 与旧 CPU 的切出收尾同步 -> 选择 CPU -> 激活。
 * 这些步骤可交错，因此这里既需要 pi_lock/rq 锁，也需要屏障。
 * success 表示本次状态唤醒被接受，不保证入队已完成或任务立即执行；
 * 仅 saved_state 匹配时也算成功，但须等锁唤醒或解冻才能真正运行。
 */
int try_to_wake_up(struct task_struct *p, unsigned int state, int wake_flags)
{
	guard(preempt)();
	int cpu, success = 0;

	wake_flags |= WF_TTWU;

	/*
	 * 除唤醒 current 的特例外，先用 pi_lock 认领这次唤醒：
	 * 仍在队列上的任务走 ttwu_runnable()，无需再次入队；
	 * 已出队的任务先转为 TASK_WAKING，再交给旧 CPU 异步处理，
	 * 或等待其完全切出后重新选 CPU 入队。
	 */
	if (p == current) {
		/*
		 * We're waking current, this means 'p->on_rq' and 'task_cpu(p)
		 * == smp_processor_id()'. Together this means we can special
		 * case the whole 'p->on_rq && ttwu_runnable()' case below
		 * without taking any locks.
		 *
		 * Specifically, given current runs ttwu() we must be before
		 * schedule()'s block_task(), as such this must not observe
		 * sched_delayed.
		 *
		 * In particular:
		 *  - we rely on Program-Order guarantees for all the ordering,
		 *  - we're serialized against set_special_state() by virtue of
		 *    it disabling IRQs (this allows not taking ->pi_lock).
		 */
		WARN_ON_ONCE(p->se.sched_delayed);
		if (!ttwu_state_match(p, state, &success))
			goto out;

		trace_sched_waking(p);
		ttwu_do_wakeup(p);
		goto out;
	}

	/*
	 * If we are going to wake up a thread waiting for CONDITION we
	 * need to ensure that CONDITION=1 done by the caller can not be
	 * reordered with p->state check below. This pairs with smp_store_mb()
	 * in set_current_state() that the waiting thread does.
	 *
	 * 等待方先设睡眠状态再检查条件，唤醒方先更新条件再检查状态。
	 * 完整屏障禁止双方同时读到旧值，否则等待方可能决定睡眠，
	 * 唤醒方却认为它仍在运行，从而丢失唤醒。
	 */
	scoped_guard (raw_spinlock_irqsave, &p->pi_lock) {
		smp_mb__after_spinlock();
		if (!ttwu_state_match(p, state, &success))
			break;

		trace_sched_waking(p);

		/*
		 * Ensure we load p->on_rq _after_ p->state, otherwise it would
		 * be possible to, falsely, observe p->on_rq == 0 and get stuck
		 * in smp_cond_load_acquire() below.
		 *
		 * sched_ttwu_pending()			try_to_wake_up()
		 *   STORE p->on_rq = 1			  LOAD p->state
		 *   UNLOCK rq->lock
		 *
		 * __schedule() (switch to task 'p')
		 *   LOCK rq->lock			  smp_rmb();
		 *   smp_mb__after_spinlock();
		 *   UNLOCK rq->lock
		 *
		 * [task p]
		 *   STORE p->state = UNINTERRUPTIBLE	  LOAD p->on_rq
		 *
		 * Pairs with the LOCK+smp_mb__after_spinlock() on rq->lock in
		 * __schedule().  See the comment for smp_mb__after_spinlock().
		 *
		 * A similar smp_rmb() lives in __task_needs_rq_lock().
		 *
		 * 必须按 state -> on_rq 的顺序读：既然看到了任务本轮请求
		 * 睡眠的状态，就不能再拿更早一次出队留下的 on_rq == 0
		 * 来判断本轮是否已出队。
		 */
		smp_rmb();
		if (READ_ONCE(p->on_rq) && ttwu_runnable(p, wake_flags))
			break;

		/*
		 * Ensure we load p->on_cpu _after_ p->on_rq, otherwise it would be
		 * possible to, falsely, observe p->on_cpu == 0.
		 *
		 * One must be running (->on_cpu == 1) in order to remove oneself
		 * from the runqueue.
		 *
		 * __schedule() (switch to task 'p')	try_to_wake_up()
		 *   STORE p->on_cpu = 1		  LOAD p->on_rq
		 *   UNLOCK rq->lock
		 *
		 * __schedule() (put 'p' to sleep)
		 *   LOCK rq->lock			  smp_rmb();
		 *   smp_mb__after_spinlock();
		 *   STORE p->on_rq = 0			  LOAD p->on_cpu
		 *
		 * Pairs with the LOCK+smp_mb__after_spinlock() on rq->lock in
		 * __schedule().  See the comment for smp_mb__after_spinlock().
		 *
		 * Form a control-dep-acquire with p->on_rq == 0 above, to ensure
		 * schedule()'s deactivate_task() has 'happened' and p will no longer
		 * care about it's own p->state. See the comment in __schedule().
		 *
		 * on_rq == 0 的分支形成控制依赖，此处将其补成 acquire，
		 * 保证后续状态访问不会越过已观察到的出队操作。
		 */
		smp_acquire__after_ctrl_dep();

		/*
		 * We're doing the wakeup (@success == 1), they did a dequeue (p->on_rq
		 * == 0), which means we need to do an enqueue, change p->state to
		 * TASK_WAKING such that we can unlock p->pi_lock before doing the
		 * enqueue, such as ttwu_queue_wakelist().
		 *
		 * TASK_WAKING 表示唤醒已被接管，但入队尚未完成，并不是
		 * TASK_RUNNING。普通唤醒者不会再匹配它，因此异步入队时
		 * 即使已释放 pi_lock，也不会把同一任务重复加入运行队列。
		 */
		WRITE_ONCE(p->__state, TASK_WAKING);

		/*
		 * If the owning (remote) CPU is still in the middle of schedule() with
		 * this task as prev, considering queueing p on the remote CPUs wake_list
		 * which potentially sends an IPI instead of spinning on p->on_cpu to
		 * let the waker make forward progress. This is safe because IRQs are
		 * disabled and the IPI will deliver after on_cpu is cleared.
		 *
		 * Ensure we load task_cpu(p) after p->on_cpu:
		 *
		 * set_task_cpu(p, cpu);
		 *   STORE p->cpu = @cpu
		 * __schedule() (switch to task 'p')
		 *   LOCK rq->lock
		 *   smp_mb__after_spin_lock()		smp_cond_load_acquire(&p->on_cpu)
		 *   STORE p->on_cpu = 1		LOAD p->cpu
		 *
		 * to ensure we observe the correct CPU on which the task is currently
		 * scheduling.
		 *
		 * 旧 CPU 正在关中断完成切出，交给它的唤醒 IPI 要等到开中断
		 * 才能处理，届时 on_cpu 已清零；唤醒方无需在此自旋。
		 */
		if (smp_load_acquire(&p->on_cpu) &&
		    ttwu_queue_wakelist(p, task_cpu(p), wake_flags))
			break;

		/*
		 * If the owning (remote) CPU is still in the middle of schedule() with
		 * this task as prev, wait until it's done referencing the task.
		 *
		 * Pairs with the smp_store_release() in finish_task().
		 *
		 * This ensures that tasks getting woken will be fully ordered against
		 * their previous state and preserve Program Order.
		 *
		 * on_rq == 0 只说明已出队，旧 CPU 可能仍在用它的内核栈。
		 * 必须等 finish_task() 以 release 清除 on_cpu，再通过这里的
		 * acquire 接手，才能安全迁移并在另一个 CPU 上重新执行。
		 */
		smp_cond_load_acquire(&p->on_cpu, !VAL);

		cpu = select_task_rq(p, p->wake_cpu, &wake_flags);
		/*
		 * 跨 CPU 唤醒要先在旧队列结束 I/O 等待及 PSI 记账，再修改
		 * task_cpu(p)。WF_MIGRATED 让目标队列识别迁移并避免重复结账。
		 */
		if (task_cpu(p) != cpu) {
			if (p->in_iowait) {
				delayacct_blkio_end(p);
				atomic_dec(&task_rq(p)->nr_iowait);
			}

			wake_flags |= WF_MIGRATED;
			psi_ttwu_dequeue(p);
			set_task_cpu(p, cpu);
		}

		ttwu_queue(p, cpu, wake_flags);
	}
out:
	if (success)
		ttwu_stat(p, task_cpu(p), wake_flags);

	return success;
}

/*
 * __task_needs_rq_lock - 持有 pi_lock 时判断任务状态
 * 是否还需要运行队列锁保护。
 * 已阻塞任务会等待切出完成，从而允许省去运行队列锁。
 */
static bool __task_needs_rq_lock(struct task_struct *p)
{
	unsigned int state = READ_ONCE(p->__state);

	/*
	 * Since pi->lock blocks try_to_wake_up(), we don't need rq->lock when
	 * the task is blocked. Make sure to check @state since ttwu() can drop
	 * locks at the end, see ttwu_queue_wakelist().
	 */
	if (state == TASK_RUNNING || state == TASK_WAKING)
		return true;

	/*
	 * Ensure we load p->on_rq after p->__state, otherwise it would be
	 * possible to, falsely, observe p->on_rq == 0.
	 *
	 * See try_to_wake_up() for a longer comment.
	 */
	smp_rmb();
	if (p->on_rq)
		return true;

	/*
	 * Ensure the task has finished __schedule() and will not be referenced
	 * anymore. Again, see try_to_wake_up() for a longer comment.
	 */
	smp_rmb();
	smp_cond_load_acquire(&p->on_cpu, !VAL);

	return false;
}

/**
 * task_call_func - Invoke a function on task in fixed state
 * 通过任务锁和必要的队列锁固定任务状态，执行轻量回调并返回其结果。
 * @p: Process for which the function is to be invoked, can be @current.
 * @func: Function to invoke.
 * @arg: Argument to function.
 *
 * Fix the task in it's current state by avoiding wakeups and or rq operations
 * and call @func(@arg) on it.  This function can use task_is_runnable() and
 * task_curr() to work out what the state is, if required.  Given that @func
 * can be invoked with a runqueue lock held, it had better be quite
 * lightweight.
 *
 * Returns:
 *   Whatever @func returns
 */
int task_call_func(struct task_struct *p, task_call_f func, void *arg)
{
	struct rq *rq = NULL;
	struct rq_flags rf;
	int ret;

	raw_spin_lock_irqsave(&p->pi_lock, rf.flags);

	/*
	 * pi_lock 只能挡住新唤醒，挡不住一个可运行任务被切换或出入队。
	 * 完全阻塞且已切出的任务才可只用 pi_lock；其他状态还须锁住 rq。
	 * 回调在关中断的锁内执行，应只做轻量检查/更新，不能睡眠。
	 */
	if (__task_needs_rq_lock(p))
		rq = __task_rq_lock(p, &rf);

	/*
	 * At this point the task is pinned; either:
	 *  - blocked and we're holding off wakeups	 (pi->lock)
	 *  - woken, and we're holding off enqueue	 (rq->lock)
	 *  - queued, and we're holding off schedule	 (rq->lock)
	 *  - running, and we're holding off de-schedule (rq->lock)
	 *
	 * The called function (@func) can use: task_curr(), p->on_rq and
	 * p->__state to differentiate between these states.
	 */
	ret = func(p, arg);

	if (rq)
		rq_unlock(rq, &rf);

	raw_spin_unlock_irqrestore(&p->pi_lock, rf.flags);
	return ret;
}

/**
 * cpu_curr_snapshot - Return a snapshot of the currently running task
 * 在运行队列锁和内存屏障保护下读取指定 CPU 的当前任务快照，
 * 不增加任务引用。
 * @cpu: The CPU on which to snapshot the task.
 *
 * Returns the task_struct pointer of the task "currently" running on
 * the specified CPU.
 *
 * If the specified CPU was offline, the return value is whatever it
 * is, perhaps a pointer to the task_struct structure of that CPU's idle
 * task, but there is no guarantee.  Callers wishing a useful return
 * value must take some action to ensure that the specified CPU remains
 * online throughout.
 *
 * This function executes full memory barriers before and after fetching
 * the pointer, which permits the caller to confine this function's fetch
 * with respect to the caller's accesses to other shared variables.
 */
struct task_struct *cpu_curr_snapshot(int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	struct task_struct *t;
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);
	smp_mb__after_spinlock(); /* Pairing determined by caller's synchronization design. */
	t = rcu_dereference(cpu_curr(cpu));
	rq_unlock_irqrestore(rq, &rf);
	smp_mb(); /* Pairing determined by caller's synchronization design. */

	return t;
}

/**
 * wake_up_process - Wake up a specific process
 * 唤醒处于普通可唤醒睡眠状态的任务，成功返回 1，否则返回 0。
 * @p: The process to be woken up.
 *
 * Attempt to wake up the nominated process and move it to the set of runnable
 * processes.
 *
 * Return: 1 if the process was woken up, 0 if it was already running.
 *
 * This function executes a full memory barrier before accessing the task state.
 */
int wake_up_process(struct task_struct *p)
{
	return try_to_wake_up(p, TASK_NORMAL, 0);
}
EXPORT_SYMBOL(wake_up_process);

/*
 * wake_up_state - 唤醒与指定状态掩码匹配的任务，返回是否成功唤醒。
 */
int wake_up_state(struct task_struct *p, unsigned int state)
{
	return try_to_wake_up(p, state, 0);
}

/*
 * Perform scheduler related setup for a newly forked process p.
 * p is forked by current.
 *
 * __sched_fork() is basic setup which is also used by sched_init() to
 * initialize the boot CPU's idle task.
 */
/*
 * __sched_fork - 初始化新任务的通用调度实体、统计、迁移及并发 ID 工作项。
 * 也用于启动阶段初始化初始空闲任务。
 */
static void __sched_fork(u64 clone_flags, struct task_struct *p)
{
	p->on_rq			= 0;

	/*
	 * p->on_rq 是核心调度层的成员状态，se.on_rq 是公平实体的
	 * 成员状态；RT、DL 等类还有自己的链接/状态。task_struct 同时
	 * 内嵌这些实体，但一次只按当前 sched_class 解释对应的调度结构。
	 */
	p->se.on_rq			= 0;
	p->se.exec_start		= 0;
	p->se.sum_exec_runtime		= 0;
	p->se.prev_sum_exec_runtime	= 0;
	p->se.nr_migrations		= 0;
	p->se.vruntime			= 0;
	p->se.vlag			= 0;
	p->se.rel_deadline		= 0;
	INIT_LIST_HEAD(&p->se.group_node);

	/* A delayed task cannot be in clone(). */
	WARN_ON_ONCE(p->se.sched_delayed);

#ifdef CONFIG_FAIR_GROUP_SCHED
	p->se.cfs_rq			= NULL;
#ifdef CONFIG_CFS_BANDWIDTH
	init_cfs_throttle_work(p);
#endif
#endif

#ifdef CONFIG_SCHEDSTATS
	/* Even if schedstat is disabled, there should not be garbage */
	memset(&p->stats, 0, sizeof(p->stats));
#endif

	init_dl_entity(&p->dl);

	INIT_LIST_HEAD(&p->rt.run_list);
	p->rt.timeout		= 0;
	p->rt.time_slice	= sched_rr_timeslice;
	p->rt.on_rq		= 0;
	p->rt.on_list		= 0;

#ifdef CONFIG_SCHED_CLASS_EXT
	init_scx_entity(&p->scx);
#endif

#ifdef CONFIG_PREEMPT_NOTIFIERS
	INIT_HLIST_HEAD(&p->preempt_notifiers);
#endif

#ifdef CONFIG_COMPACTION
	p->capture_control = NULL;
#endif
	init_numa_balancing(clone_flags, p);
	p->wake_entry.u_flags = CSD_TYPE_TTWU;
	p->migration_pending = NULL;
	init_sched_mm_cid(p);
}

DEFINE_STATIC_KEY_FALSE(sched_numa_balancing);

#ifdef CONFIG_NUMA_BALANCING

int sysctl_numa_balancing_mode;

/*
 * __set_numabalancing_state - 通过静态分支启用或禁用 NUMA 自动均衡快速路径。
 */
static void __set_numabalancing_state(bool enabled)
{
	if (enabled)
		static_branch_enable(&sched_numa_balancing);
	else
		static_branch_disable(&sched_numa_balancing);
}

/*
 * set_numabalancing_state - 设置 NUMA 均衡的普通或禁用模式，并同步静态分支。
 */
void set_numabalancing_state(bool enabled)
{
	if (enabled)
		sysctl_numa_balancing_mode = NUMA_BALANCING_NORMAL;
	else
		sysctl_numa_balancing_mode = NUMA_BALANCING_DISABLED;
	__set_numabalancing_state(enabled);
}

#ifdef CONFIG_PROC_SYSCTL
/*
 * reset_memory_tiering - 重置在线内存节点的分层迁移阈值及候选页统计基准。
 */
static void reset_memory_tiering(void)
{
	struct pglist_data *pgdat;

	for_each_online_pgdat(pgdat) {
		pgdat->nbp_threshold = 0;
		pgdat->nbp_th_nr_cand = node_page_state(pgdat, PGPROMOTE_CANDIDATE);
		pgdat->nbp_th_start = jiffies_to_msecs(jiffies);
	}
}

/*
 * sysctl_numa_balancing - 处理 NUMA 均衡模式的 sysctl 读写，
 * 并在启用内存分层时重置统计。
 * 写入操作要求 CAP_SYS_ADMIN 权限。
 */
static int sysctl_numa_balancing(const struct ctl_table *table, int write,
			  void *buffer, size_t *lenp, loff_t *ppos)
{
	struct ctl_table t;
	int err;
	int state = sysctl_numa_balancing_mode;

	if (write && !capable(CAP_SYS_ADMIN))
		return -EPERM;

	t = *table;
	t.data = &state;
	err = proc_dointvec_minmax(&t, write, buffer, lenp, ppos);
	if (err < 0)
		return err;
	if (write) {
		if (!(sysctl_numa_balancing_mode & NUMA_BALANCING_MEMORY_TIERING) &&
		    (state & NUMA_BALANCING_MEMORY_TIERING))
			reset_memory_tiering();
		sysctl_numa_balancing_mode = state;
		__set_numabalancing_state(state);
	}
	return err;
}
#endif /* CONFIG_PROC_SYSCTL */
#endif /* CONFIG_NUMA_BALANCING */

#ifdef CONFIG_SCHEDSTATS

DEFINE_STATIC_KEY_FALSE(sched_schedstats);

/*
 * set_schedstats - 切换调度统计静态分支，控制详细统计的运行时开销。
 */
static void set_schedstats(bool enabled)
{
	if (enabled)
		static_branch_enable(&sched_schedstats);
	else
		static_branch_disable(&sched_schedstats);
}

/*
 * force_schedstat_enabled - 在性能分析需要时强制开启调度统计，
 * 并提示管理员。
 */
void force_schedstat_enabled(void)
{
	if (!schedstat_enabled()) {
		pr_info("kernel profiling enabled schedstats, disable via kernel.sched_schedstats.\n");
		static_branch_enable(&sched_schedstats);
	}
}

/*
 * setup_schedstats - 解析 schedstats 启动参数中的 enable 或 disable，
 * 返回处理状态。
 */
static int __init setup_schedstats(char *str)
{
	int ret = 0;
	if (!str)
		goto out;

	if (!strcmp(str, "enable")) {
		set_schedstats(true);
		ret = 1;
	} else if (!strcmp(str, "disable")) {
		set_schedstats(false);
		ret = 1;
	}
out:
	if (!ret)
		pr_warn("Unable to parse schedstats=\n");

	return ret;
}
__setup("schedstats=", setup_schedstats);

#ifdef CONFIG_PROC_SYSCTL
/*
 * sysctl_schedstats - 处理调度统计开关的 sysctl 访问，
 * 校验权限和取值后更新静态分支。
 */
static int sysctl_schedstats(const struct ctl_table *table, int write, void *buffer,
		size_t *lenp, loff_t *ppos)
{
	struct ctl_table t;
	int err;
	int state = static_branch_likely(&sched_schedstats);

	if (write && !capable(CAP_SYS_ADMIN))
		return -EPERM;

	t = *table;
	t.data = &state;
	err = proc_dointvec_minmax(&t, write, buffer, lenp, ppos);
	if (err < 0)
		return err;
	if (write)
		set_schedstats(state);
	return err;
}
#endif /* CONFIG_PROC_SYSCTL */
#endif /* CONFIG_SCHEDSTATS */

#ifdef CONFIG_SYSCTL
static const struct ctl_table sched_core_sysctls[] = {
#ifdef CONFIG_SCHEDSTATS
	{
		.procname       = "sched_schedstats",
		.data           = NULL,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = sysctl_schedstats,
		.extra1         = SYSCTL_ZERO,
		.extra2         = SYSCTL_ONE,
	},
#endif /* CONFIG_SCHEDSTATS */
#ifdef CONFIG_UCLAMP_TASK
	{
		.procname       = "sched_util_clamp_min",
		.data           = &sysctl_sched_uclamp_util_min,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = sysctl_sched_uclamp_handler,
	},
	{
		.procname       = "sched_util_clamp_max",
		.data           = &sysctl_sched_uclamp_util_max,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = sysctl_sched_uclamp_handler,
	},
	{
		.procname       = "sched_util_clamp_min_rt_default",
		.data           = &sysctl_sched_uclamp_util_min_rt_default,
		.maxlen         = sizeof(unsigned int),
		.mode           = 0644,
		.proc_handler   = sysctl_sched_uclamp_handler,
	},
#endif /* CONFIG_UCLAMP_TASK */
#ifdef CONFIG_NUMA_BALANCING
	{
		.procname	= "numa_balancing",
		.data		= NULL, /* filled in by handler */
		.maxlen		= sizeof(unsigned int),
		.mode		= 0644,
		.proc_handler	= sysctl_numa_balancing,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_FOUR,
	},
#endif /* CONFIG_NUMA_BALANCING */
};
/*
 * sched_core_sysctl_init - 注册核心调度相关的 kernel sysctl 配置项。
 */
static int __init sched_core_sysctl_init(void)
{
	register_sysctl_init("kernel", sched_core_sysctls);
	return 0;
}
late_initcall(sched_core_sysctl_init);
#endif /* CONFIG_SYSCTL */

/*
 * fork()/clone()-time setup:
 */
/*
 * sched_fork - 初始化子任务的调度策略、优先级和统计，
 * 避免继承临时优先级提升。
 * 将任务保持为 TASK_NEW；不允许的截止时间策略继承会返回 -EAGAIN。
 */
int sched_fork(u64 clone_flags, struct task_struct *p)
{
	/*
	 * fork 已复制父 task_struct，这里必须清除父任务的队列链接、
	 * 累计运行时间和在 CPU 状态。继承策略与新建运行实体是两件事，
	 * 不能把父任务当前的调度位置或 PI 临时提升一并继承。
	 */
	__sched_fork(clone_flags, p);
	/*
	 * We mark the process as NEW here. This guarantees that
	 * nobody will actually run it, and a signal or other external
	 * event cannot wake it up and insert it on the runqueue either.
	 *
	 * TASK_NEW 隔离尚未完成初始化的子任务；普通唤醒不会使其提前
	 * 入队，首次变为可运行由 wake_up_new_task() 专门处理。
	 */
	p->__state = TASK_NEW;

	/*
	 * Make sure we do not leak PI boosting priority to the child.
	 */
	p->prio = current->normal_prio;

	uclamp_fork(p);

	/*
	 * Revert to default priority/policy on fork if requested.
	 */
	if (unlikely(p->sched_reset_on_fork)) {
		if (task_has_dl_policy(p) || task_has_rt_policy(p)) {
			p->policy = SCHED_NORMAL;
			p->static_prio = NICE_TO_PRIO(0);
			p->rt_priority = 0;
			p->timer_slack_ns = p->default_timer_slack_ns;
		} else if (PRIO_TO_NICE(p->static_prio) < 0)
			p->static_prio = NICE_TO_PRIO(0);

		p->prio = p->normal_prio = p->static_prio;
		set_load_weight(p, false);
		p->se.custom_slice = 0;
		p->se.slice = sysctl_sched_base_slice;

		/*
		 * We don't need the reset flag anymore after the fork. It has
		 * fulfilled its duty:
		 */
		p->sched_reset_on_fork = 0;
	}

	/* DL 继承若未被 reset-on-fork 化解，此路径拒绝创建，未复制预算。 */
	if (dl_prio(p->prio))
		return -EAGAIN;

	scx_pre_fork(p);

	if (rt_prio(p->prio)) {
		p->sched_class = &rt_sched_class;
#ifdef CONFIG_SCHED_CLASS_EXT
	} else if (task_should_scx(p->policy)) {
		p->sched_class = &ext_sched_class;
#endif
	} else {
		p->sched_class = &fair_sched_class;
	}

	init_entity_runnable_average(&p->se);


#ifdef CONFIG_SCHED_INFO
	if (likely(sched_info_on()))
		memset(&p->sched_info, 0, sizeof(p->sched_info));
#endif
	p->on_cpu = 0;
	init_task_preempt_count(p);
	plist_node_init(&p->pushable_tasks, MAX_PRIO);
	RB_CLEAR_NODE(&p->pushable_dl_tasks);

	return 0;
}

/*
 * sched_cgroup_fork - 为新任务关联调度任务组、设置初始 CPU，
 * 并执行调度类 fork 回调。
 */
int sched_cgroup_fork(struct task_struct *p, struct kernel_clone_args *kargs)
{
	unsigned long flags;

	/*
	 * Because we're not yet on the pid-hash, p->pi_lock isn't strictly
	 * required yet, but lockdep gets upset if rules are violated.
	 */
	raw_spin_lock_irqsave(&p->pi_lock, flags);
#ifdef CONFIG_CGROUP_SCHED
	if (1) {
		struct task_group *tg;
		tg = container_of(kargs->cset->subsys[cpu_cgrp_id],
				  struct task_group, css);
		tg = autogroup_task_group(p, tg);
		p->sched_task_group = tg;
	}
#endif
	rseq_migrate(p);
	/*
	 * We're setting the CPU for the first time, we don't migrate,
	 * so use __set_task_cpu().
	 */
	__set_task_cpu(p, smp_processor_id());
	if (p->sched_class->task_fork)
		p->sched_class->task_fork(p);
	raw_spin_unlock_irqrestore(&p->pi_lock, flags);

	return scx_fork(p);
}

/*
 * sched_cancel_fork - 创建任务失败时撤销 sched_ext 已完成的 fork 准备。
 */
void sched_cancel_fork(struct task_struct *p)
{
	scx_cancel_fork(p);
}

/*
 * sched_post_fork - 在任务创建后完成利用率钳制及 sched_ext 的后续初始化。
 */
void sched_post_fork(struct task_struct *p)
{
	uclamp_post_fork(p);
	scx_post_fork(p);
}

/*
 * to_ratio - 将运行时间与周期换算为定点带宽比例，
 * 无限运行时间返回满带宽单位。
 */
u64 to_ratio(u64 period, u64 runtime)
{
	if (runtime == RUNTIME_INF)
		return BW_UNIT;

	/*
	 * Doing this here saves a lot of checks in all
	 * the calling paths, and returning zero seems
	 * safe for them anyway.
	 */
	if (period == 0)
		return 0;

	return div64_u64(runtime << BW_SHIFT, period);
}

/*
 * wake_up_new_task - wake up a newly created task for the first time.
 *
 * This function will do some initial scheduler statistics housekeeping
 * that must be done for every newly created context, then puts the task
 * on the runqueue and wakes it.
 */
/*
 * wake_up_new_task - 首次为新任务选择 CPU、初始化利用率并入队，
 * 检查其抢占需求。
 */
void wake_up_new_task(struct task_struct *p)
{
	struct rq_flags rf;
	struct rq *rq;
	int wake_flags = WF_FORK;

	raw_spin_lock_irqsave(&p->pi_lock, rf.flags);
	WRITE_ONCE(p->__state, TASK_RUNNING);
	/*
	 * Fork balancing, do it here and not earlier because:
	 *  - cpus_ptr can change in the fork path
	 *  - any previously selected CPU might disappear through hotplug
	 *
	 * Use __set_task_cpu() to avoid calling sched_class::migrate_task_rq,
	 * as we're not fully set-up yet.
	 *
	 * 首次入队时才最终选择 CPU，使用创建流程结束后的亲和性和在线
	 * 状态；这里不是已运行任务的迁移，不能套用普通迁移回调。
	 */
	p->recent_used_cpu = task_cpu(p);
	rseq_migrate(p);
	__set_task_cpu(p, select_task_rq(p, task_cpu(p), &wake_flags));
	rq = __task_rq_lock(p, &rf);
	update_rq_clock(rq);
	post_init_entity_util_avg(p);

	/*
	 * INITIAL 让调度类执行首次放置规则；这不属于睡眠后的 WAKEUP，
	 * 无需结束旧的阻塞记账。首次入队同样只提出抢占请求。
	 */
	activate_task(rq, p, ENQUEUE_NOCLOCK | ENQUEUE_INITIAL);
	trace_sched_wakeup_new(p);
	wakeup_preempt(rq, p, wake_flags);
	if (p->sched_class->task_woken) {
		/*
		 * Nothing relies on rq->lock after this, so it's fine to
		 * drop it.
		 */
		rq_unpin_lock(rq, &rf);
		p->sched_class->task_woken(rq, p);
		rq_repin_lock(rq, &rf);
	}
	task_rq_unlock(rq, p, &rf);
}

#ifdef CONFIG_PREEMPT_NOTIFIERS

static DEFINE_STATIC_KEY_FALSE(preempt_notifier_key);

/*
 * preempt_notifier_inc - 增加抢占通知功能的使用计数，
 * 必要时开启通知静态分支。
 */
void preempt_notifier_inc(void)
{
	static_branch_inc(&preempt_notifier_key);
}
EXPORT_SYMBOL_GPL(preempt_notifier_inc);

/*
 * preempt_notifier_dec - 减少抢占通知功能的使用计数，
 * 最后一个使用者退出时关闭分支。
 */
void preempt_notifier_dec(void)
{
	static_branch_dec(&preempt_notifier_key);
}
EXPORT_SYMBOL_GPL(preempt_notifier_dec);

/**
 * preempt_notifier_register - tell me when current is being preempted & rescheduled
 * 为当前任务注册切入和切出通知，要求通知功能已启用。
 * @notifier: notifier struct to register
 */
void preempt_notifier_register(struct preempt_notifier *notifier)
{
	if (!static_branch_unlikely(&preempt_notifier_key))
		WARN(1, "registering preempt_notifier while notifiers disabled\n");

	hlist_add_head(&notifier->link, &current->preempt_notifiers);
}
EXPORT_SYMBOL_GPL(preempt_notifier_register);

/**
 * preempt_notifier_unregister - no longer interested in preemption notifications
 * 注销任务的抢占通知项，不可在抢占通知回调内部调用。
 * @notifier: notifier struct to unregister
 *
 * This is *not* safe to call from within a preemption notifier.
 */
void preempt_notifier_unregister(struct preempt_notifier *notifier)
{
	hlist_del(&notifier->link);
}
EXPORT_SYMBOL_GPL(preempt_notifier_unregister);

/*
 * __fire_sched_in_preempt_notifiers - 遍历任务的通知链表，调用切入 CPU 回调。
 */
static void __fire_sched_in_preempt_notifiers(struct task_struct *curr)
{
	struct preempt_notifier *notifier;

	hlist_for_each_entry(notifier, &curr->preempt_notifiers, link)
		notifier->ops->sched_in(notifier, raw_smp_processor_id());
}

/*
 * fire_sched_in_preempt_notifiers - 仅在通知静态分支启用时发出任务切入通知。
 */
static __always_inline void fire_sched_in_preempt_notifiers(struct task_struct *curr)
{
	if (static_branch_unlikely(&preempt_notifier_key))
		__fire_sched_in_preempt_notifiers(curr);
}

/*
 * __fire_sched_out_preempt_notifiers - 遍历任务的通知链表。
 * 通知当前任务即将被 next 替换。
 */
static void
__fire_sched_out_preempt_notifiers(struct task_struct *curr,
				   struct task_struct *next)
{
	struct preempt_notifier *notifier;

	hlist_for_each_entry(notifier, &curr->preempt_notifiers, link)
		notifier->ops->sched_out(notifier, next);
}

/*
 * fire_sched_out_preempt_notifiers - 仅在通知静态分支启用时发出任务切出通知。
 */
static __always_inline void
fire_sched_out_preempt_notifiers(struct task_struct *curr,
				 struct task_struct *next)
{
	if (static_branch_unlikely(&preempt_notifier_key))
		__fire_sched_out_preempt_notifiers(curr, next);
}

#else /* !CONFIG_PREEMPT_NOTIFIERS: */

/*
 * fire_sched_in_preempt_notifiers - 未启用抢占通知时的空操作切入接口。
 */
static inline void fire_sched_in_preempt_notifiers(struct task_struct *curr)
{
}

/*
 * fire_sched_out_preempt_notifiers - 未启用抢占通知时的空操作切出接口。
 */
static inline void
fire_sched_out_preempt_notifiers(struct task_struct *curr,
				 struct task_struct *next)
{
}

#endif /* !CONFIG_PREEMPT_NOTIFIERS */

/*
 * prepare_task - 在切入任务之前发布 on_cpu 标志，与并发唤醒路径协调。
 */
static inline void prepare_task(struct task_struct *next)
{
	/*
	 * 此时 prev 尚未清除 on_cpu，所以短暂出现两个 on_cpu==1
	 * 是正常交接窗口，不能据此推断同一逻辑 CPU 同时执行两个任务。
	 */
	/*
	 * Claim the task as running, we do this before switching to it
	 * such that any running task will have this set.
	 *
	 * See the smp_load_acquire(&p->on_cpu) case in ttwu() and
	 * its ordering comment.
	 */
	WRITE_ONCE(next->on_cpu, 1);
}

/*
 * finish_task - 在本 CPU 最后一次访问切出任务后，
 * 以释放语义清除其 on_cpu 标志。
 */
static inline void finish_task(struct task_struct *prev)
{
	/*
	 * This must be the very last reference to @prev from this CPU. After
	 * p->on_cpu is cleared, the task can be moved to a different CPU. We
	 * must ensure this doesn't happen until the switch is completely
	 * finished.
	 *
	 * In particular, the load of prev->state in finish_task_switch() must
	 * happen before this.
	 *
	 * Pairs with the smp_cond_load_acquire() in try_to_wake_up().
	 *
	 * 这是切出方交出任务的发布点，而不是普通布尔赋值：唤醒方一旦
	 * 读到零，就可能让任务在别的 CPU 上运行，故必须发布此前的收尾
	 * 操作，且后面不能再依赖该任务仍由本 CPU 独占。
	 */
	smp_store_release(&prev->on_cpu, 0);
}

/*
 * do_balance_callbacks - 持有运行队列锁时逐个摘除并执行均衡回调。
 */
static void do_balance_callbacks(struct rq *rq, struct balance_callback *head)
{
	void (*func)(struct rq *rq);
	struct balance_callback *next;

	lockdep_assert_rq_held(rq);

	while (head) {
		/*
		 * 先摘除链接，允许回调重新排队；回调可能暂时放下 rq 锁，
		 * 因此不能在复杂的调度类属性更新中间直接执行它。
		 */
		func = (void (*)(struct rq *))head->func;
		next = head->next;
		head->next = NULL;
		head = next;

		func(rq);
	}
}

static void balance_push(struct rq *rq);

/*
 * balance_push_callback is a right abuse of the callback interface and plays
 * by significantly different rules.
 *
 * Where the normal balance_callback's purpose is to be ran in the same context
 * that queued it (only later, when it's safe to drop rq->lock again),
 * balance_push_callback is specifically targeted at __schedule().
 *
 * This abuse is tolerated because it places all the unlikely/odd cases behind
 * a single test, namely: rq->balance_callback == NULL.
 */
struct balance_callback balance_push_callback = {
	.next = NULL,
	.func = balance_push,
};

/*
 * __splice_balance_callbacks - 摘取运行队列的均衡回调链表，
 * 分段处理时保留热拔出回调。
 */
static inline struct balance_callback *
__splice_balance_callbacks(struct rq *rq, bool split)
{
	struct balance_callback *head = rq->balance_callback;

	if (likely(!head))
		return NULL;

	lockdep_assert_rq_held(rq);
	/*
	 * Must not take balance_push_callback off the list when
	 * splice_balance_callbacks() and balance_callbacks() are not
	 * in the same rq->lock section.
	 *
	 * In that case it would be possible for __schedule() to interleave
	 * and observe the list empty.
	 */
	if (split && head == &balance_push_callback)
		head = NULL;
	else
		rq->balance_callback = NULL;

	return head;
}

/*
 * splice_balance_callbacks - 获取可在另一持锁区间执行的回调链表，
 * 保留持久推送回调。
 */
struct balance_callback *splice_balance_callbacks(struct rq *rq)
{
	return __splice_balance_callbacks(rq, true);
}

/*
 * __balance_callbacks - 在已持有运行队列锁的上下文摘取并执行全部均衡回调。
 */
static void __balance_callbacks(struct rq *rq)
{
	do_balance_callbacks(rq, __splice_balance_callbacks(rq, false));
}

/*
 * balance_callbacks - 保存中断状态并获取运行队列锁，
 * 执行此前摘出的均衡回调。
 */
void balance_callbacks(struct rq *rq, struct balance_callback *head)
{
	unsigned long flags;

	if (unlikely(head)) {
		raw_spin_rq_lock_irqsave(rq, flags);
		do_balance_callbacks(rq, head);
		raw_spin_rq_unlock_irqrestore(rq, flags);
	}
}

/*
 * prepare_lock_switch - 为跨任务交接运行队列锁
 * 更新 lockdep 和调试锁所有者信息。
 */
static inline void
prepare_lock_switch(struct rq *rq, struct task_struct *next, struct rq_flags *rf)
{
	/*
	 * Since the runqueue lock will be released by the next
	 * task (which is an invalid locking op but in the case
	 * of the scheduler it's an obvious special-case), so we
	 * do an early lockdep release here:
	 *
	 * 此处只交接锁的调试记账，并未真正解锁。实际 rq 锁会跨越
	 * switch_to()，由切入任务在 finish_lock_switch() 中释放，
	 * 从而使选任务、换栈和 on_cpu 状态交接处于同一个临界区。
	 */
	rq_unpin_lock(rq, rf);
	spin_release(&__rq_lockp(rq)->dep_map, _THIS_IP_);
#ifdef CONFIG_DEBUG_SPINLOCK
	/* this is a valid case when another task releases the spinlock */
	rq_lockp(rq)->owner = next;
#endif
}

/*
 * finish_lock_switch - 在新任务上下文恢复锁依赖记账，
 * 执行均衡回调并解锁、开中断。
 */
static inline void finish_lock_switch(struct rq *rq)
{
	/*
	 * If we are tracking spinlock dependencies then we have to
	 * fix up the runqueue lock - which gets 'carried over' from
	 * prev into current:
	 */
	spin_acquire(&__rq_lockp(rq)->dep_map, 0, 0, _THIS_IP_);
	/*
	 * 普通的“取锁者必须亲自解锁”在此有意被调度器打破。若在
	 * switch_to() 前真正放锁，远端就可能重排或迁移尚未切完的任务；
	 * 调试工具的 release/acquire 标注是在描述交接，并非缩短临界区。
	 */
	__balance_callbacks(rq);
	raw_spin_rq_unlock_irq(rq);
}

/*
 * NOP if the arch has not defined these:
 */

#ifndef prepare_arch_switch
# define prepare_arch_switch(next)	do { } while (0)
#endif

#ifndef finish_arch_post_lock_switch
# define finish_arch_post_lock_switch()	do { } while (0)
#endif

/*
 * kmap_local_sched_out - 任务切出时保存或撤销其活动的
 * 本地临时高端内存映射。
 */
static inline void kmap_local_sched_out(void)
{
#ifdef CONFIG_KMAP_LOCAL
	if (unlikely(current->kmap_ctrl.idx))
		__kmap_local_sched_out();
#endif
}

/*
 * kmap_local_sched_in - 任务切入时恢复其本地临时高端内存映射。
 */
static inline void kmap_local_sched_in(void)
{
#ifdef CONFIG_KMAP_LOCAL
	if (unlikely(current->kmap_ctrl.idx))
		__kmap_local_sched_in();
#endif
}

/**
 * prepare_task_switch - prepare to switch tasks
 * 在关中断且持有队列锁时准备上下文切换，
 * 更新统计并调用架构和通知钩子。
 * @rq: the runqueue preparing to switch
 * @prev: the current task that is being switched out
 * @next: the task we are going to switch to.
 *
 * This is called with the rq lock held and interrupts off. It must
 * be paired with a subsequent finish_task_switch after the context
 * switch.
 *
 * prepare_task_switch sets up locking and calls architecture specific
 * hooks.
 */
static inline void
prepare_task_switch(struct rq *rq, struct task_struct *prev,
		    struct task_struct *next)
{
	/*
	 * 仍在 prev 的执行上下文：先结束性能/调度统计和 rseq 临界区，
	 * 再通知依赖 CPU 状态的使用者、撤下本地映射，最后认领 next。
	 * 这些钩子处理的对象不同，不应与稍后的寄存器切换混为一谈。
	 */
	kcov_prepare_switch(prev);
	sched_info_switch(rq, prev, next);
	perf_event_task_sched_out(prev, next);
	rseq_preempt(prev);
	fire_sched_out_preempt_notifiers(prev, next);
	kmap_local_sched_out();
	prepare_task(next);
	prepare_arch_switch(next);
}

/**
 * finish_task_switch - clean up after a task-switch
 * 在切入任务中完成切换收尾、解锁、延迟地址空间释放，
 * 以及死亡任务资源回收。
 * @prev: the thread we just switched away from.
 *
 * finish_task_switch must be called after the context switch, paired
 * with a prepare_task_switch call before the context switch.
 * finish_task_switch will reconcile locking set up by prepare_task_switch,
 * and do any other architecture-specific cleanup actions.
 *
 * Note that we may have delayed dropping an mm in context_switch(). If
 * so, we finish that here outside of the runqueue lock. (Doing it
 * with the lock held can cause deadlocks; see schedule() for
 * details.)
 *
 * The context switch have flipped the stack from under us and restored the
 * local variables which were saved when this task called schedule() in the
 * past. 'prev == current' is still correct but we need to recalculate this_rq
 * because prev may have moved to another CPU.
 */
static struct rq *finish_task_switch(struct task_struct *prev)
	__releases(rq->lock)
{
	struct rq *rq = this_rq();
	struct mm_struct *mm = rq->prev_mm;
	unsigned int prev_state;

	/*
	 * 已在切入任务的栈上。局部变量来自该任务上次切出时保存的栈，
	 * 它可能已迁到另一个 CPU，所以重新取 this_rq()；传入的 prev
	 * 则是本次刚切出的任务，而非本任务上次让出 CPU 时的对手。
	 */
	/*
	 * The previous task will have left us with a preempt_count of 2
	 * because it left us after:
	 *
	 *	schedule()
	 *	  preempt_disable();			// 1
	 *	  __schedule()
	 *	    raw_spin_lock_irq(&rq->lock)	// 2
	 *
	 * Also, see FORK_PREEMPT_COUNT.
	 */
	if (WARN_ONCE(preempt_count() != 2*PREEMPT_DISABLE_OFFSET,
		      "corrupted preempt_count: %s/%d/0x%x\n",
		      current->comm, current->pid, preempt_count()))
		preempt_count_set(FORK_PREEMPT_COUNT);

	rq->prev_mm = NULL;

	/*
	 * A task struct has one reference for the use as "current".
	 * If a task dies, then it sets TASK_DEAD in tsk->state and calls
	 * schedule one last time. The schedule call will never return, and
	 * the scheduled task must drop that reference.
	 *
	 * We must observe prev->state before clearing prev->on_cpu (in
	 * finish_task), otherwise a concurrent wakeup can get prev
	 * running on another CPU and we could rave with its RUNNING -> DEAD
	 * transition, resulting in a double drop.
	 *
	 * 必须先保存本次切出的状态，再由 finish_task() 清除 on_cpu。
	 * 否则 prev 可能已在别的 CPU 上再次运行甚至退出，本 CPU 若读到
	 * 后一次退出的 TASK_DEAD，就可能与那次切换重复释放任务引用。
	 */
	prev_state = READ_ONCE(prev->__state);
	vtime_task_switch(prev);
	perf_event_task_sched_in(prev, current);
	finish_task(prev);
	tick_nohz_task_switch();
	finish_lock_switch(rq);
	finish_arch_post_lock_switch();
	kcov_finish_switch(current);
	/*
	 * kmap_local_sched_out() is invoked with rq::lock held and
	 * interrupts disabled. There is no requirement for that, but the
	 * sched out code does not have an interrupt enabled section.
	 * Restoring the maps on sched in does not require interrupts being
	 * disabled either.
	 */
	kmap_local_sched_in();

	/*
	 * Any cached block-layer timestamp (plug->cur_ktime) is stale now,
	 * invalidate it.
	 */
	blk_plug_invalidate_ts();

	fire_sched_in_preempt_notifiers(current);
	/*
	 * When switching through a kernel thread, the loop in
	 * membarrier_{private,global}_expedited() may have observed that
	 * kernel thread and not issued an IPI. It is therefore possible to
	 * schedule between user->kernel->user threads without passing though
	 * switch_mm(). Membarrier requires a barrier after storing to
	 * rq->curr, before returning to userspace, so provide them here:
	 *
	 * - a full memory barrier for {PRIVATE,GLOBAL}_EXPEDITED, implicitly
	 *   provided by mmdrop_lazy_tlb(),
	 * - a sync_core for SYNC_CORE.
	 */
	if (mm) {
		membarrier_mm_sync_core_before_usermode(mm);
		mmdrop_lazy_tlb_sched(mm);
	}

	if (unlikely(prev_state == TASK_DEAD)) {
		/*
		 * 死亡任务不能在自己的栈上释放自己的栈；由接替者在换栈后
		 * 完成，task_struct 的最终回收再遵守 RCU 读者的生命周期。
		 */
		if (prev->sched_class->task_dead)
			prev->sched_class->task_dead(prev);

		/* Task is done with its stack. */
		put_task_stack(prev);

		put_task_struct_rcu_user(prev);
	}

	return rq;
}

/**
 * schedule_tail - first thing a freshly forked thread must call.
 * 新任务首次运行时完成上下文切换收尾，
 * 恢复抢占并设置子线程 ID 和待处理信号。
 * @prev: the thread we just switched away from.
 */
asmlinkage __visible void schedule_tail(struct task_struct *prev)
	__releases(rq->lock)
{
	/*
	 * 新任务没有“上次从 switch_to 返回”的历史栈帧，由架构的 fork
	 * 返回入口走到这里，补做普通任务在 context_switch 尾部完成的收尾。
	 */
	/*
	 * New tasks start with FORK_PREEMPT_COUNT, see there and
	 * finish_task_switch() for details.
	 *
	 * finish_task_switch() will drop rq->lock() and lower preempt_count
	 * and the preempt_enable() will end up enabling preemption (on
	 * PREEMPT_COUNT kernels).
	 */

	finish_task_switch(prev);
	/*
	 * This is a special case: the newly created task has just
	 * switched the context for the first time. It is returning from
	 * schedule for the first time in this path.
	 */
	trace_sched_exit_tp(true);
	preempt_enable();

	if (current->set_child_tid)
		put_user(task_pid_vnr(current), current->set_child_tid);

	calculate_sigpending();
}

/*
 * context_switch - switch to the new MM and the new thread's register state.
 */
/*
 * context_switch - 切换地址空间、并发 ID、寄存器和内核栈，
 * 并完成切换后清理。
 * 调用时运行队列已加锁且中断关闭，返回时队列锁已释放。
 */
static __always_inline struct rq *
context_switch(struct rq *rq, struct task_struct *prev,
	       struct task_struct *next, struct rq_flags *rf)
{
	/*
	 * 三阶段：准备状态 -> 切地址空间/寄存器/内核栈 -> 新栈上收尾。
	 * rq 锁贯穿前两阶段，直至 finish_task_switch() 才真正释放。
	 * 此函数“返回”时 current 已是获得 CPU 的任务，不能把它当成
	 * 普通函数调用后仍在原任务上继续顺序执行。
	 */
	prepare_task_switch(rq, prev, next);

	/*
	 * For paravirt, this is coupled with an exit in switch_to to
	 * combine the page table reload and the switch backend into
	 * one hypercall.
	 */
	arch_start_context_switch(prev);

	/*
	 * kernel -> kernel   lazy + transfer active
	 *   user -> kernel   lazy + mmgrab_lazy_tlb() active
	 *
	 * kernel ->   user   switch + mmdrop_lazy_tlb() active
	 *   user ->   user   switch
	 *
	 * switch_mm_cid() needs to be updated if the barriers provided
	 * by context_switch() are modified.
	 *
	 * mm 是任务自身的用户地址空间，内核线程的 mm 为 NULL；
	 * active_mm 则记录 CPU 正在使用或借用的地址空间。切入内核线程
	 * 可借用切出任务的 active_mm，避免为执行内核代码切换页表。
	 *
	 * 可用“地址空间引用由谁持有”检查下面四种分支：
	 *   用户任务 -> 内核线程：借用原 active_mm，按配置取得借用引用；
	 *   内核线程 -> 内核线程：转交借用关系，清空 prev 的 active_mm；
	 *   内核线程 -> 用户任务：切到 next->mm，借用引用延后归还；
	 *   用户任务 -> 用户任务：按体系结构规则切换到 next->mm。
	 * “用户任务”指 mm 非空，切换时它仍在内核态；这里不是直接
	 * 返回用户态；同一地址空间内的线程切换也不一定要重装页表。
	 * 借用引用的增减由 mmgrab/mmdrop_lazy_tlb 辅助函数按配置处理。
	 */
	if (!next->mm) {                                // to kernel
		enter_lazy_tlb(prev->active_mm, next);

		next->active_mm = prev->active_mm;
		if (prev->mm)                           // from user
			mmgrab_lazy_tlb(prev->active_mm);
		else
			prev->active_mm = NULL;
	} else {                                        // to user
		membarrier_switch_mm(rq, prev->active_mm, next->mm);
		/*
		 * sys_membarrier() requires an smp_mb() between setting
		 * rq->curr / membarrier_switch_mm() and returning to userspace.
		 *
		 * The below provides this either through switch_mm(), or in
		 * case 'prev->active_mm == next->mm' through
		 * finish_task_switch()'s mmdrop().
		 */
		switch_mm_irqs_off(prev->active_mm, next->mm, next);
		lru_gen_use_mm(next->mm);

		if (!prev->mm) {                        // from kernel
			/*
			 * will mmdrop_lazy_tlb() in finish_task_switch().
			 *
			 * 借用的 mm 暂存到 rq，切换完成并解锁后再放引用，
			 * 避免持有调度器锁时释放地址空间而死锁。
			 */
			rq->prev_mm = prev->active_mm;
			prev->active_mm = NULL;
		}
	}

	/* switch_mm_cid() requires the memory barriers above. */
	switch_mm_cid(rq, prev, next);

	prepare_lock_switch(rq, next, rf);

	/*
	 * Here we just switch the register state and the stack.
	 * 这里不是在原栈上调用 next：寄存器和内核栈都会切换。当前任务
	 * 日后再次获得 CPU 时，才从自己保存的 switch_to() 位置继续；
	 * 第三个参数接收这次恢复前刚被切出的任务，供后续收尾使用。
	 */
	switch_to(prev, next, prev);
	/* 编译器屏障约束切换前后的访存优化，不替代跨 CPU 的内存屏障。 */
	barrier();

	return finish_task_switch(prev);
}

/*
 * nr_running and nr_context_switches:
 *
 * externally visible scheduler statistics: current number of runnable
 * threads, total number of context switches performed since bootup.
 */
/*
 * nr_running - 汇总所有在线 CPU 的可运行任务数量，结果为非原子统计快照。
 */
unsigned int nr_running(void)
{
	unsigned int i, sum = 0;

	/*
	 * 没有同时锁住所有 rq，求和期间任务可唤醒、阻塞或迁移，所以
	 * 这是观察性统计。不能据此证明全系统在某一瞬间恰有 sum 个任务，
	 * 更不能把它作为后续操作无需加锁的依据。
	 */
	for_each_online_cpu(i)
		sum += cpu_rq(i)->nr_running;

	return sum;
}

/*
 * Check if only the current task is running on the CPU.
 *
 * Caution: this function does not check that the caller has disabled
 * preemption, thus the result might have a time-of-check-to-time-of-use
 * race.  The caller is responsible to use it correctly, for example:
 *
 * - from a non-preemptible section (of course)
 *
 * - from a thread that is bound to a single CPU
 *
 * - in a loop with very short iterations (e.g. a polling loop)
 */
/*
 * single_task_running - 判断本 CPU 是否仅有一个可运行任务。
 * 调用者须自行防范抢占或迁移造成的检查后状态变化。
 */
bool single_task_running(void)
{
	return raw_rq()->nr_running == 1;
}
EXPORT_SYMBOL(single_task_running);

/*
 * nr_context_switches_cpu - 读取指定 CPU 自启动以来累计的上下文切换次数。
 */
unsigned long long nr_context_switches_cpu(int cpu)
{
	return cpu_rq(cpu)->nr_switches;
}

/*
 * nr_context_switches - 汇总所有可能 CPU 的上下文切换次数，
 * 包含已离线 CPU 的历史值。
 */
unsigned long long nr_context_switches(void)
{
	int i;
	unsigned long long sum = 0;

	for_each_possible_cpu(i)
		sum += cpu_rq(i)->nr_switches;

	return sum;
}

/*
 * Consumers of these two interfaces, like for example the cpuidle menu
 * governor, are using nonsensical data. Preferring shallow idle state selection
 * for a CPU that has IO-wait which might not even end up running the task when
 * it does become runnable.
 */

/*
 * nr_iowait_cpu - 读取记账在指定 CPU 上的 I/O 等待任务数，
 * 不代表严格的 CPU 归属。
 */
unsigned int nr_iowait_cpu(int cpu)
{
	return atomic_read(&cpu_rq(cpu)->nr_iowait);
}

/*
 * IO-wait accounting, and how it's mostly bollocks (on SMP).
 *
 * The idea behind IO-wait account is to account the idle time that we could
 * have spend running if it were not for IO. That is, if we were to improve the
 * storage performance, we'd have a proportional reduction in IO-wait time.
 *
 * This all works nicely on UP, where, when a task blocks on IO, we account
 * idle time as IO-wait, because if the storage were faster, it could've been
 * running and we'd not be idle.
 *
 * This has been extended to SMP, by doing the same for each CPU. This however
 * is broken.
 *
 * Imagine for instance the case where two tasks block on one CPU, only the one
 * CPU will have IO-wait accounted, while the other has regular idle. Even
 * though, if the storage were faster, both could've ran at the same time,
 * utilising both CPUs.
 *
 * This means, that when looking globally, the current IO-wait accounting on
 * SMP is a lower bound, by reason of under accounting.
 *
 * Worse, since the numbers are provided per CPU, they are sometimes
 * interpreted per CPU, and that is nonsensical. A blocked task isn't strictly
 * associated with any one particular CPU, it can wake to another CPU than it
 * blocked on. This means the per CPU IO-wait number is meaningless.
 *
 * Task CPU affinities can make all that even more 'interesting'.
 */

/*
 * nr_iowait - 汇总所有可能 CPU 记录的 I/O 等待任务数量。
 */
unsigned int nr_iowait(void)
{
	unsigned int i, sum = 0;

	for_each_possible_cpu(i)
		sum += nr_iowait_cpu(i);

	return sum;
}

/*
 * sched_exec - execve() is a valuable balancing opportunity, because at
 * this point the task has the smallest effective memory and cache footprint.
 */
/*
 * sched_exec - 利用 exec 时缓存足迹较小的机会重新选核，
 * 并通过 stopper 迁移当前任务。
 */
void sched_exec(void)
{
	struct task_struct *p = current;
	struct migration_arg arg;
	int dest_cpu;

	scoped_guard (raw_spinlock_irqsave, &p->pi_lock) {
		dest_cpu = p->sched_class->select_task_rq(p, task_cpu(p), WF_EXEC);
		if (dest_cpu == smp_processor_id())
			return;

		if (unlikely(!cpu_active(dest_cpu)))
			return;

		arg = (struct migration_arg){ p, dest_cpu };
	}
	stop_one_cpu(task_cpu(p), migration_cpu_stop, &arg);
}

DEFINE_PER_CPU(struct kernel_stat, kstat);
DEFINE_PER_CPU(struct kernel_cpustat, kernel_cpustat);

EXPORT_PER_CPU_SYMBOL(kstat);
EXPORT_PER_CPU_SYMBOL(kernel_cpustat);

/*
 * The function fair_sched_class.update_curr accesses the struct curr
 * and its field curr->exec_start; when called from task_sched_runtime(),
 * we observe a high rate of cache misses in practice.
 * Prefetching this data results in improved performance.
 */
/*
 * prefetch_curr_exec_start - 预取公平调度当前实体及其执行起点，
 * 降低运行时间查询开销。
 */
static inline void prefetch_curr_exec_start(struct task_struct *p)
{
#ifdef CONFIG_FAIR_GROUP_SCHED
	struct sched_entity *curr = p->se.cfs_rq->curr;
#else
	struct sched_entity *curr = task_rq(p)->cfs.curr;
#endif
	prefetch(curr);
	prefetch(&curr->exec_start);
}

/*
 * Return accounted runtime for the task.
 * In case the task is currently running, return the runtime plus current's
 * pending runtime that have not been accounted yet.
 */
/*
 * task_sched_runtime - 返回任务累计执行时间，
 * 必要时补记尚未结算的当前执行片段。
 * 返回值单位为纳秒。
 */
unsigned long long task_sched_runtime(struct task_struct *p)
{
	struct rq_flags rf;
	struct rq *rq;
	u64 ns;

#ifdef CONFIG_64BIT
	/*
	 * 64-bit doesn't need locks to atomically read a 64-bit value.
	 * So we have a optimization chance when the task's delta_exec is 0.
	 * Reading ->on_cpu is racy, but this is OK.
	 *
	 * If we race with it leaving CPU, we'll take a lock. So we're correct.
	 * If we race with it entering CPU, unaccounted time is 0. This is
	 * indistinguishable from the read occurring a few cycles earlier.
	 * If we see ->on_cpu without ->on_rq, the task is leaving, and has
	 * been accounted, so we're correct here as well.
	 */
	if (!p->on_cpu || !task_on_rq_queued(p))
		return p->se.sum_exec_runtime;
#endif

	rq = task_rq_lock(p, &rf);
	/*
	 * Must be ->curr _and_ ->on_rq.  If dequeued, we would
	 * project cycles that may never be accounted to this
	 * thread, breaking clock_gettime().
	 */
	if (task_current_donor(rq, p) && task_on_rq_queued(p)) {
		/*
		 * sum_exec_runtime 是已结算值；当前实体还欠从 exec_start 到
		 * 此刻的一段时间，先由调度类 update_curr 结算再返回。
		 */
		prefetch_curr_exec_start(p);
		update_rq_clock(rq);
		p->sched_class->update_curr(rq);
	}
	ns = p->se.sum_exec_runtime;
	task_rq_unlock(rq, p, &rf);

	return ns;
}

/*
 * cpu_resched_latency - 跟踪本 CPU 重调度请求的等待时间，
 * 超出告警阈值时返回纳秒延迟。
 * 未超阈值或不应告警时返回 0。
 */
static u64 cpu_resched_latency(struct rq *rq)
{
	int latency_warn_ms = READ_ONCE(sysctl_resched_latency_warn_ms);
	u64 resched_latency, now = rq_clock(rq);
	static bool warned_once;

	if (sysctl_resched_latency_warn_once && warned_once)
		return 0;

	if (!need_resched() || !latency_warn_ms)
		return 0;

	if (system_state == SYSTEM_BOOTING)
		return 0;

	if (!rq->last_seen_need_resched_ns) {
		rq->last_seen_need_resched_ns = now;
		rq->ticks_without_resched = 0;
		return 0;
	}

	rq->ticks_without_resched++;
	resched_latency = now - rq->last_seen_need_resched_ns;
	if (resched_latency <= latency_warn_ms * NSEC_PER_MSEC)
		return 0;

	warned_once = true;

	return resched_latency;
}

/*
 * setup_resched_latency_warn_ms - 解析启动参数，
 * 设置重调度延迟告警阈值，单位为毫秒。
 */
static int __init setup_resched_latency_warn_ms(char *str)
{
	long val;

	if ((kstrtol(str, 0, &val))) {
		pr_warn("Unable to set resched_latency_warn_ms\n");
		return 1;
	}

	sysctl_resched_latency_warn_ms = val;
	return 1;
}
__setup("resched_latency_warn_ms=", setup_resched_latency_warn_ms);

/*
 * This function gets called by the timer code, with HZ frequency.
 * We call it with interrupts disabled.
 */
/*
 * sched_tick - 在周期定时中断中更新调度记账、任务时间片和负载，
 * 并触发必要的均衡。
 * 调用时中断必须关闭。
 */
void sched_tick(void)
{
	int cpu = smp_processor_id();
	struct rq *rq = cpu_rq(cpu);
	/* scheduler accounting goes to the donor task */
	struct task_struct *curr, *donor;
	struct rq_flags rf;
	unsigned long hw_pressure;
	u64 resched_latency;

	if (housekeeping_cpu(cpu, HK_TYPE_KERNEL_NOISE))
		arch_scale_freq_tick();

	sched_clock_tick();

	rq_lock(rq, &rf);
	curr = rq->curr;
	donor = rq->donor;

	psi_account_irqtime(rq, curr, NULL);

	update_rq_clock(rq);
	hw_pressure = arch_scale_hw_pressure(cpu_of(rq));
	update_hw_load_avg(rq_clock_task(rq), rq, hw_pressure);

	/* 延迟请求遇到周期 tick 升为立即请求，推动重新调度。 */
	if (dynamic_preempt_lazy() && tif_test_bit(TIF_NEED_RESCHED_LAZY))
		resched_curr(rq);

	/*
	 * 周期 tick 是记账和策略检查点，具体时间片规则在调度类中。
	 * 代理执行按 donor 的调度资源记账，执行者 curr 仍用于工作线程
	 * 等执行上下文通知；两者在未启用代理执行时为同一任务。
	 */
	donor->sched_class->task_tick(rq, donor, 0);
	if (sched_feat(LATENCY_WARN))
		resched_latency = cpu_resched_latency(rq);
	calc_global_load_tick(rq);
	sched_core_tick(rq);
	task_tick_mm_cid(rq, donor);
	scx_tick(rq);

	rq_unlock(rq, &rf);

	if (sched_feat(LATENCY_WARN) && resched_latency)
		resched_latency_warn(cpu, resched_latency);

	perf_event_task_tick();

	if (curr->flags & PF_WQ_WORKER)
		wq_worker_tick(curr);

	if (!scx_switched_all()) {
		/* 到期均衡由调度软中断推进，避免在 tick 中遍历迁移任务。 */
		rq->idle_balance = idle_cpu(cpu);
		sched_balance_trigger(rq);
	}
}

#ifdef CONFIG_NO_HZ_FULL

struct tick_work {
	int			cpu;
	atomic_t		state;
	struct delayed_work	work;
};
/* Values for ->state, see diagram below. */
#define TICK_SCHED_REMOTE_OFFLINE	0
#define TICK_SCHED_REMOTE_OFFLINING	1
#define TICK_SCHED_REMOTE_RUNNING	2

/*
 * 这里的状态描述远程工作项是否应继续排队，而非 CPU 的 online 位：
 * RUNNING 时回调自行排下一次；stop 仅改成 OFFLINING，由在途回调
 * 降到 OFFLINE 后停止续排。若此时又 start，改回 RUNNING 即可复用
 * 在途工作；只有从 OFFLINE 启动才新排一个，避免同一工作重复启动。
 */

/*
 * State diagram for ->state:
 *
 *
 *          TICK_SCHED_REMOTE_OFFLINE
 *                    |   ^
 *                    |   |
 *                    |   | sched_tick_remote()
 *                    |   |
 *                    |   |
 *                    +--TICK_SCHED_REMOTE_OFFLINING
 *                    |   ^
 *                    |   |
 * sched_tick_start() |   | sched_tick_stop()
 *                    |   |
 *                    V   |
 *          TICK_SCHED_REMOTE_RUNNING
 *
 *
 * Other transitions get WARN_ON_ONCE(), except that sched_tick_remote()
 * and sched_tick_start() are happy to leave the state in RUNNING.
 */

static struct tick_work __percpu *tick_work_cpu;

/*
 * sched_tick_remote - 为停止本地 tick 的远端 CPU 执行低频调度记账，
 * 并维护卸载状态机。
 */
static void sched_tick_remote(struct work_struct *work)
{
	struct delayed_work *dwork = to_delayed_work(work);
	struct tick_work *twork = container_of(dwork, struct tick_work, work);
	int cpu = twork->cpu;
	struct rq *rq = cpu_rq(cpu);
	int os;

	/*
	 * NO_HZ_FULL 停掉的是目标 CPU 的频繁本地 tick；housekeeping
	 * 工作线程仍可低频锁住其 rq，结算运行时间并维护全局负载。
	 * 需要周期轮转的场景不由此处理，能否停 tick 已另行检查。
	 */
	/*
	 * Handle the tick only if it appears the remote CPU is running in full
	 * dynticks mode. The check is racy by nature, but missing a tick or
	 * having one too much is no big deal because the scheduler tick updates
	 * statistics and checks timeslices in a time-independent way, regardless
	 * of when exactly it is running.
	 */
	if (tick_nohz_tick_stopped_cpu(cpu)) {
		guard(rq_lock_irq)(rq);
		struct task_struct *curr = rq->curr;

		if (cpu_online(cpu)) {
			/*
			 * Since this is a remote tick for full dynticks mode,
			 * we are always sure that there is no proxy (only a
			 * single task is running).
			 */
			WARN_ON_ONCE(rq->curr != rq->donor);
			update_rq_clock(rq);

			if (!is_idle_task(curr)) {
				/*
				 * Make sure the next tick runs within a
				 * reasonable amount of time.
				 */
				u64 delta = rq_clock_task(rq) - curr->se.exec_start;
				WARN_ON_ONCE(delta > (u64)NSEC_PER_SEC * 3);
			}
			curr->sched_class->task_tick(rq, curr, 0);

			calc_load_nohz_remote(rq);
		}
	}

	/*
	 * Run the remote tick once per second (1Hz). This arbitrary
	 * frequency is large enough to avoid overload but short enough
	 * to keep scheduler internal stats reasonably up to date.  But
	 * first update state to reflect hotplug activity if required.
	 */
	os = atomic_fetch_add_unless(&twork->state, -1, TICK_SCHED_REMOTE_RUNNING);
	WARN_ON_ONCE(os == TICK_SCHED_REMOTE_OFFLINE);
	if (os == TICK_SCHED_REMOTE_RUNNING)
		queue_delayed_work(system_unbound_wq, dwork, HZ);
}

/*
 * sched_tick_start - 为非 housekeeping CPU 启动每秒一次的远程调度 tick 工作。
 */
static void sched_tick_start(int cpu)
{
	int os;
	struct tick_work *twork;

	if (housekeeping_cpu(cpu, HK_TYPE_KERNEL_NOISE))
		return;

	WARN_ON_ONCE(!tick_work_cpu);

	twork = per_cpu_ptr(tick_work_cpu, cpu);
	os = atomic_xchg(&twork->state, TICK_SCHED_REMOTE_RUNNING);
	WARN_ON_ONCE(os == TICK_SCHED_REMOTE_RUNNING);
	if (os == TICK_SCHED_REMOTE_OFFLINE) {
		twork->cpu = cpu;
		INIT_DELAYED_WORK(&twork->work, sched_tick_remote);
		queue_delayed_work(system_unbound_wq, &twork->work, HZ);
	}
}

#ifdef CONFIG_HOTPLUG_CPU
/*
 * sched_tick_stop - 将远程 tick 标记为正在停止，由工作项完成后续状态转换。
 */
static void sched_tick_stop(int cpu)
{
	struct tick_work *twork;
	int os;

	if (housekeeping_cpu(cpu, HK_TYPE_KERNEL_NOISE))
		return;

	WARN_ON_ONCE(!tick_work_cpu);

	twork = per_cpu_ptr(tick_work_cpu, cpu);
	/* There cannot be competing actions, but don't rely on stop-machine. */
	os = atomic_xchg(&twork->state, TICK_SCHED_REMOTE_OFFLINING);
	WARN_ON_ONCE(os != TICK_SCHED_REMOTE_RUNNING);
	/* Don't cancel, as this would mess up the state machine. */
}
#endif /* CONFIG_HOTPLUG_CPU */

/*
 * sched_tick_offload_init - 分配各 CPU 的远程调度 tick 工作状态，
 * 失败时终止启动。
 */
int __init sched_tick_offload_init(void)
{
	tick_work_cpu = alloc_percpu(struct tick_work);
	BUG_ON(!tick_work_cpu);
	return 0;
}

#else /* !CONFIG_NO_HZ_FULL: */
/*
 * sched_tick_start - 未启用全动态 tick 时的空操作启动接口。
 */
static inline void sched_tick_start(int cpu) { }
/*
 * sched_tick_stop - 未启用全动态 tick 时的空操作停止接口。
 */
static inline void sched_tick_stop(int cpu) { }
#endif /* !CONFIG_NO_HZ_FULL */

#if defined(CONFIG_PREEMPTION) && (defined(CONFIG_DEBUG_PREEMPT) || \
				defined(CONFIG_TRACE_PREEMPT_TOGGLE))
/*
 * If the value passed in is equal to the current preempt count
 * then we just disabled preemption. Start timing the latency.
 */
/*
 * preempt_latency_start - 进入最外层禁止抢占区间时
 * 记录调用位置并开始延迟跟踪。
 */
static inline void preempt_latency_start(int val)
{
	if (preempt_count() == val) {
		unsigned long ip = get_lock_parent_ip();
#ifdef CONFIG_DEBUG_PREEMPT
		current->preempt_disable_ip = ip;
#endif
		trace_preempt_off(CALLER_ADDR0, ip);
	}
}

/*
 * preempt_count_add - 增加抢占嵌套计数，
 * 检查异常并在必要时启动抢占延迟跟踪。
 */
void preempt_count_add(int val)
{
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Underflow?
	 */
	if (DEBUG_LOCKS_WARN_ON((preempt_count() < 0)))
		return;
#endif
	__preempt_count_add(val);
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Spinlock count overflowing soon?
	 */
	DEBUG_LOCKS_WARN_ON((preempt_count() & PREEMPT_MASK) >=
				PREEMPT_MASK - 10);
#endif
	preempt_latency_start(val);
}
EXPORT_SYMBOL(preempt_count_add);
NOKPROBE_SYMBOL(preempt_count_add);

/*
 * If the value passed in equals to the current preempt count
 * then we just enabled preemption. Stop timing the latency.
 */
/*
 * preempt_latency_stop - 即将退出最外层禁止抢占区间时结束抢占延迟跟踪。
 */
static inline void preempt_latency_stop(int val)
{
	if (preempt_count() == val)
		trace_preempt_on(CALLER_ADDR0, get_lock_parent_ip());
}

/*
 * preempt_count_sub - 检查并减少抢占嵌套计数，退出最外层前结束延迟跟踪。
 */
void preempt_count_sub(int val)
{
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Underflow?
	 */
	if (DEBUG_LOCKS_WARN_ON(val > preempt_count()))
		return;
	/*
	 * Is the spinlock portion underflowing?
	 */
	if (DEBUG_LOCKS_WARN_ON((val < PREEMPT_MASK) &&
			!(preempt_count() & PREEMPT_MASK)))
		return;
#endif

	preempt_latency_stop(val);
	__preempt_count_sub(val);
}
EXPORT_SYMBOL(preempt_count_sub);
NOKPROBE_SYMBOL(preempt_count_sub);

#else
/*
 * preempt_latency_start - 未启用抢占延迟跟踪时的空操作起始接口。
 */
static inline void preempt_latency_start(int val) { }
/*
 * preempt_latency_stop - 未启用抢占延迟跟踪时的空操作结束接口。
 */
static inline void preempt_latency_stop(int val) { }
#endif

/*
 * get_preempt_disable_ip - 获取任务最近禁止抢占的位置，未启用调试时返回 0。
 */
static inline unsigned long get_preempt_disable_ip(struct task_struct *p)
{
#ifdef CONFIG_DEBUG_PREEMPT
	return p->preempt_disable_ip;
#else
	return 0;
#endif
}

/*
 * Print scheduling while atomic bug:
 */
/*
 * __schedule_bug - 报告原子上下文中调度的错误，
 * 输出锁、抢占位置和调用栈等诊断信息。
 */
static noinline void __schedule_bug(struct task_struct *prev)
{
	/* Save this before calling printk(), since that will clobber it */
	unsigned long preempt_disable_ip = get_preempt_disable_ip(current);

	if (oops_in_progress)
		return;

	printk(KERN_ERR "BUG: scheduling while atomic: %s/%d/0x%08x\n",
		prev->comm, prev->pid, preempt_count());

	debug_show_held_locks(prev);
	print_modules();
	if (irqs_disabled())
		print_irqtrace_events(prev);
	if (IS_ENABLED(CONFIG_DEBUG_PREEMPT)) {
		pr_err("Preemption disabled at:");
		print_ip_sym(KERN_ERR, preempt_disable_ip);
	}
	check_panic_on_warn("scheduling while atomic");

	dump_stack();
	add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
}

/*
 * Various schedule()-time debugging checks and statistics:
 */
/*
 * schedule_debug - 调度前检查栈、原子状态和 RCU 约束，并更新调度调试统计。
 */
static inline void schedule_debug(struct task_struct *prev, bool preempt)
{
#ifdef CONFIG_SCHED_STACK_END_CHECK
	if (task_stack_end_corrupted(prev))
		panic("corrupted stack end detected inside scheduler\n");

	if (task_scs_end_corrupted(prev))
		panic("corrupted shadow stack detected inside scheduler\n");
#endif

#ifdef CONFIG_DEBUG_ATOMIC_SLEEP
	if (!preempt && READ_ONCE(prev->__state) && prev->non_block_count) {
		printk(KERN_ERR "BUG: scheduling in a non-blocking section: %s/%d/%i\n",
			prev->comm, prev->pid, prev->non_block_count);
		dump_stack();
		add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
	}
#endif

	if (unlikely(in_atomic_preempt_off())) {
		__schedule_bug(prev);
		preempt_count_set(PREEMPT_DISABLED);
	}
	rcu_sleep_check();
	WARN_ON_ONCE(ct_state() == CT_STATE_USER);

	profile_hit(SCHED_PROFILING, __builtin_return_address(0));

	schedstat_inc(this_rq()->sched_count);
}

/*
 * prev_balance - 在放下前一个调度任务之前，
 * 按调度类优先级执行选取前的负载均衡。
 */
static void prev_balance(struct rq *rq, struct rq_flags *rf)
{
	const struct sched_class *start_class = rq->donor->sched_class;
	const struct sched_class *class;

#ifdef CONFIG_SCHED_CLASS_EXT
	/*
	 * SCX requires a balance() call before every pick_task() including when
	 * waking up from SCHED_IDLE. If @start_class is below SCX, start from
	 * SCX instead. Also, set a flag to detect missing balance() call.
	 */
	if (scx_enabled()) {
		rq->scx.flags |= SCX_RQ_BAL_PENDING;
		if (sched_class_above(&ext_sched_class, start_class))
			start_class = &ext_sched_class;
	}
#endif

	/*
	 * We must do the balancing pass before put_prev_task(), such
	 * that when we release the rq->lock the task is in the same
	 * state as before we took rq->lock.
	 *
	 * We can terminate the balance pass as soon as we know there is
	 * a runnable task of @class priority or higher.
	 */
	for_active_class_range(class, start_class, &idle_sched_class) {
		if (class->balance && class->balance(rq, rf))
			break;
	}
}

/*
 * Pick up the highest-prio task:
 */
/*
 * __pick_next_task - 按调度类优先级选出下一任务，
 * 并优化仅有公平类任务的常见路径。
 */
static inline struct task_struct *
__pick_next_task(struct rq *rq, struct rq_flags *rf)
{
	const struct sched_class *class;
	struct task_struct *p;

	rq->dl_server = NULL;

	/*
	 * 各调度类的优先级由类排列决定，不按所有任务的 prio 混排。
	 * fair 类内部再使用自身的公平性/虚拟截止时间规则；不能将下面
	 * 的公平类快路径理解为按一个全局 vruntime 树选择全系统任务。
	 */
	if (scx_enabled())
		goto restart;

	/*
	 * Optimization: we know that if all tasks are in the fair class we can
	 * call that function directly, but only if the @prev task wasn't of a
	 * higher scheduling class, because otherwise those lose the
	 * opportunity to pull in more work from other CPUs.
	 */
	if (likely(!sched_class_above(rq->donor->sched_class, &fair_sched_class) &&
		   rq->nr_running == rq->cfs.h_nr_queued)) {

		p = pick_next_task_fair(rq, rq->donor, rf);
		/*
		 * 公平类的空闲均衡可能暂时释放 rq 锁，期间更高优先级任务
		 * 可以入队；RETRY_TASK 要求重新遍历调度类，不能直接选 idle。
		 */
		if (unlikely(p == RETRY_TASK))
			goto restart;

		/* Assume the next prioritized class is idle_sched_class */
		if (!p) {
			p = pick_task_idle(rq);
			put_prev_set_next_task(rq, rq->donor, p);
		}

		return p;
	}

restart:
	prev_balance(rq, rf);

	/*
	 * 按启用调度类的优先级从高到低遍历，首个候选即为结果。
	 * pick_next_task() 包含前后任务的调度类状态交接；只有 pick_task()
	 * 的类则由核心补做 put_prev_set_next_task()。idle 类负责最终兜底。
	 */
	for_each_active_class(class) {
		if (class->pick_next_task) {
			p = class->pick_next_task(rq, rq->donor);
			if (p)
				return p;
		} else {
			p = class->pick_task(rq);
			if (p) {
				put_prev_set_next_task(rq, rq->donor, p);
				return p;
			}
		}
	}

	BUG(); /* The idle class should always have a runnable task. */
}

#ifdef CONFIG_SCHED_CORE
/*
 * is_task_rq_idle - 判断任务是否为其运行队列专属的空闲任务。
 */
static inline bool is_task_rq_idle(struct task_struct *t)
{
	return (task_rq(t)->idle == t);
}

/*
 * cookie_equals - 判断任务是否匹配指定核心 cookie，空闲任务视为总是匹配。
 */
static inline bool cookie_equals(struct task_struct *a, unsigned long cookie)
{
	return is_task_rq_idle(a) || (a->core_cookie == cookie);
}

/*
 * cookie_match - 判断两个任务能否在同一核心并行运行，
 * 任一为空闲任务时允许匹配。
 */
static inline bool cookie_match(struct task_struct *a, struct task_struct *b)
{
	if (is_task_rq_idle(a) || is_task_rq_idle(b))
		return true;

	return a->core_cookie == b->core_cookie;
}

/*
 * pick_task - 按调度类优先级取得候选任务，
 * 不执行前后任务的运行状态交接。
 */
static inline struct task_struct *pick_task(struct rq *rq)
{
	const struct sched_class *class;
	struct task_struct *p;

	rq->dl_server = NULL;

	for_each_active_class(class) {
		p = class->pick_task(rq);
		if (p)
			return p;
	}

	BUG(); /* The idle class should always have a runnable task. */
}

extern void task_vruntime_update(struct rq *rq, struct task_struct *p, bool in_fi);

static void queue_core_balance(struct rq *rq);

/*
 * pick_next_task - 为同一 SMT 核心协调选择 cookie 兼容的任务，
 * 必要时强制部分 CPU 空闲。
 * 维护核心级选择缓存、强制空闲记账及兄弟 CPU 重调度请求。
 */
static struct task_struct *
pick_next_task(struct rq *rq, struct rq_flags *rf)
{
	struct task_struct *next, *p, *max = NULL;
	const struct cpumask *smt_mask;
	bool fi_before = false;
	bool core_clock_updated = (rq == rq->core);
	unsigned long cookie;
	int i, cpu, occ = 0;
	struct rq *rq_i;
	bool need_sync;

	/*
	 * 整核选择协调同一物理核的 SMT 逻辑 CPU，而非合并它们的队列。
	 * 启用后这些 rq 共用实际锁，选择者可在一个临界区内检查兄弟 rq；
	 * 每个 CPU 最终仍从自己的队列切入自己的 core_pick。
	 */
	if (!sched_core_enabled(rq))
		return __pick_next_task(rq, rf);

	cpu = cpu_of(rq);

	/* Stopper task is switching into idle, no need core-wide selection. */
	if (cpu_is_offline(cpu)) {
		/*
		 * Reset core_pick so that we don't enter the fastpath when
		 * coming online. core_pick would already be migrated to
		 * another cpu during offline.
		 */
		rq->core_pick = NULL;
		rq->core_dl_server = NULL;
		return __pick_next_task(rq, rf);
	}

	/*
	 * 某些调度类选任务时会暂时释放 rq 锁；此计数让核心调度开关
	 * 等待整轮选择结束，避免在选取期间重绑 rq 对应的实际锁。
	 */
	rq->core->core_pick_in_flight++;

	/*
	 * If there were no {en,de}queues since we picked (IOW, the task
	 * pointers are all still valid), and we haven't scheduled the last
	 * pick yet, do so now.
	 *
	 * rq->core_pick can be NULL if no selection was made for a CPU because
	 * it was either offline or went offline during a sibling's core-wide
	 * selection. In this case, do a core-wide selection.
	 *
	 * 一次核心级选择会给所有 SMT 兄弟缓存候选任务，各 CPU 随后消费。
	 * pick_seq == task_seq 表示候选集未因入队/出队或新一轮选择失效；
	 * pick_seq != core_sched_seq 表示本 CPU 还没消费这轮结果。
	 * 两者同时成立且缓存非空，才可以跳过核心级重新选择。
	 */
	if (rq->core->core_pick_seq == rq->core->core_task_seq &&
	    rq->core->core_pick_seq != rq->core_sched_seq &&
	    rq->core_pick) {
		WRITE_ONCE(rq->core_sched_seq, rq->core->core_pick_seq);

		next = rq->core_pick;
		rq->dl_server = rq->core_dl_server;
		rq->core_pick = NULL;
		rq->core_dl_server = NULL;
		goto out_set_next;
	}

	prev_balance(rq, rf);

	smt_mask = cpu_smt_mask(cpu);
	need_sync = !!rq->core->core_cookie;

	/* reset state */
	rq->core->core_cookie = 0UL;
	if (rq->core->core_forceidle_count) {
		if (!core_clock_updated) {
			update_rq_clock(rq->core);
			core_clock_updated = true;
		}
		sched_core_account_forceidle(rq);
		/* reset after accounting force idle */
		rq->core->core_forceidle_start = 0;
		rq->core->core_forceidle_count = 0;
		rq->core->core_forceidle_occupation = 0;
		need_sync = true;
		fi_before = true;
	}

	/*
	 * core->core_task_seq, core->core_pick_seq, rq->core_sched_seq
	 *
	 * @task_seq guards the task state ({en,de}queues)
	 * @pick_seq is the @task_seq we did a selection on
	 * @sched_seq is the @pick_seq we scheduled
	 *
	 * However, preemptions can cause multiple picks on the same task set.
	 * 'Fix' this by also increasing @task_seq for every pick.
	 */
	rq->core->core_task_seq++;

	/*
	 * Optimize for common case where this CPU has no cookies
	 * and there are no cookied tasks running on siblings.
	 */
	if (!need_sync) {
		next = pick_task(rq);
		if (!next->core_cookie) {
			rq->core_pick = NULL;
			rq->core_dl_server = NULL;
			/*
			 * For robustness, update the min_vruntime_fi for
			 * unconstrained picks as well.
			 */
			WARN_ON_ONCE(fi_before);
			task_vruntime_update(rq, next, false);
			goto out_set_next;
		}
	}

	/*
	 * For each thread: do the regular task pick and find the max prio task
	 * amongst them.
	 *
	 * Tie-break prio towards the current CPU
	 *
	 * 第一轮暂不施加 cookie 约束，先比较各兄弟 CPU 的最优候选，
	 * 由整个核心中优先级最高的任务决定本轮允许并行运行的 cookie，
	 * 避免低优先级任务反过来约束高优先级任务。
	 */
	for_each_cpu_wrap(i, smt_mask, cpu) {
		rq_i = cpu_rq(i);

		/*
		 * Current cpu always has its clock updated on entrance to
		 * pick_next_task(). If the current cpu is not the core,
		 * the core may also have been updated above.
		 */
		if (i != cpu && (rq_i != rq->core || !core_clock_updated))
			update_rq_clock(rq_i);

		rq_i->core_pick = p = pick_task(rq_i);
		rq_i->core_dl_server = rq_i->dl_server;

		if (!max || prio_less(max, p, fi_before))
			max = p;
	}

	cookie = rq->core->core_cookie = max->core_cookie;

	/*
	 * For each thread: try and find a runnable task that matches @max or
	 * force idle.
	 *
	 * 第二轮保留匹配任务，替换不匹配的候选；找不到同 cookie 的任务
	 * 就选 idle。这会牺牲部分并行度，但不能让互不信任的任务仅因
	 * 位于不同逻辑 CPU 就同时使用同一物理核心的共享资源。
	 */
	for_each_cpu(i, smt_mask) {
		rq_i = cpu_rq(i);
		p = rq_i->core_pick;

		if (!cookie_equals(p, cookie)) {
			p = NULL;
			if (cookie)
				p = sched_core_find(rq_i, cookie);
			if (!p)
				p = idle_sched_class.pick_task(rq_i);
		}

		rq_i->core_pick = p;
		rq_i->core_dl_server = NULL;

		if (p == rq_i->idle) {
			/* 有任务却因 cookie 约束不能执行，才计为强制空闲。 */
			if (rq_i->nr_running) {
				rq->core->core_forceidle_count++;
				if (!fi_before)
					rq->core->core_forceidle_seq++;
			}
		} else {
			occ++;
		}
	}

	if (schedstat_enabled() && rq->core->core_forceidle_count) {
		rq->core->core_forceidle_start = rq_clock(rq->core);
		rq->core->core_forceidle_occupation = occ;
	}

	rq->core->core_pick_seq = rq->core->core_task_seq;
	next = rq->core_pick;
	rq->core_sched_seq = rq->core->core_pick_seq;

	/* Something should have been selected for current CPU */
	WARN_ON_ONCE(!next);

	/*
	 * Reschedule siblings
	 *
	 * NOTE: L1TF -- at this point we're no longer running the old task and
	 * sending an IPI (below) ensures the sibling will no longer be running
	 * their task. This ensures there is no inter-sibling overlap between
	 * non-matching user state.
	 */
	for_each_cpu(i, smt_mask) {
		rq_i = cpu_rq(i);

		/*
		 * An online sibling might have gone offline before a task
		 * could be picked for it, or it might be offline but later
		 * happen to come online, but its too late and nothing was
		 * picked for it.  That's Ok - it will pick tasks for itself,
		 * so ignore it.
		 */
		if (!rq_i->core_pick)
			continue;

		/*
		 * Update for new !FI->FI transitions, or if continuing to be in !FI:
		 * fi_before     fi      update?
		 *  0            0       1
		 *  0            1       1
		 *  1            0       1
		 *  1            1       0
		 */
		if (!(fi_before && rq->core->core_forceidle_count))
			task_vruntime_update(rq_i, rq_i->core_pick, !!rq->core->core_forceidle_count);

		rq_i->core_pick->core_occupation = occ;

		if (i == cpu) {
			rq_i->core_pick = NULL;
			rq_i->core_dl_server = NULL;
			continue;
		}

		/* Did we break L1TF mitigation requirements? */
		WARN_ON_ONCE(!cookie_match(next, rq_i->core_pick));

		if (rq_i->curr == rq_i->core_pick) {
			rq_i->core_pick = NULL;
			rq_i->core_dl_server = NULL;
			continue;
		}

		resched_curr(rq_i);
	}

out_set_next:
	rq->core->core_pick_in_flight--;
	put_prev_set_next_task(rq, rq->donor, next);
	if (rq->core->core_forceidle_count && next == rq->idle)
		queue_core_balance(rq);

	return next;
}

/*
 * try_steal_cookie - 从另一 CPU 拉取 cookie 匹配且允许迁移的任务，
 * 缓解强制空闲。
 * 成功迁移返回 true。
 */
static bool try_steal_cookie(int this, int that)
{
	struct rq *dst = cpu_rq(this), *src = cpu_rq(that);
	struct task_struct *p;
	unsigned long cookie;
	bool success = false;

	guard(irq)();
	guard(double_rq_lock)(dst, src);

	cookie = dst->core->core_cookie;
	if (!cookie)
		return false;

	if (dst->curr != dst->idle)
		return false;

	p = sched_core_find(src, cookie);
	if (!p)
		return false;

	do {
		if (p == src->core_pick || p == src->curr)
			goto next;

		if (!is_cpu_allowed(p, this))
			goto next;

		if (p->core_occupation > dst->idle->core_occupation)
			goto next;
		/*
		 * sched_core_find() and sched_core_next() will ensure
		 * that task @p is not throttled now, we also need to
		 * check whether the runqueue of the destination CPU is
		 * being throttled.
		 */
		if (sched_task_is_throttled(p, this))
			goto next;

		move_queued_task_locked(src, dst, p);
		resched_curr(dst);

		success = true;
		break;

next:
		p = sched_core_next(p, cookie);
	} while (p);

	return success;
}

/*
 * steal_cookie_task - 在指定调度域中寻找可为目标 CPU 拉取的 cookie 匹配任务。
 */
static bool steal_cookie_task(int cpu, struct sched_domain *sd)
{
	int i;

	for_each_cpu_wrap(i, sched_domain_span(sd), cpu + 1) {
		if (i == cpu)
			continue;

		if (need_resched())
			break;

		if (try_steal_cookie(cpu, i))
			return true;
	}

	return false;
}

/*
 * sched_core_balance - 暂时释放本队列锁，
 * 逐级搜索调度域以拉取兼容核心 cookie 的任务。
 * 返回前重新获取队列锁并恢复关中断状态。
 */
static void sched_core_balance(struct rq *rq)
{
	struct sched_domain *sd;
	int cpu = cpu_of(rq);

	guard(preempt)();
	guard(rcu)();

	raw_spin_rq_unlock_irq(rq);
	for_each_domain(cpu, sd) {
		if (need_resched())
			break;

		if (steal_cookie_task(cpu, sd))
			break;
	}
	raw_spin_rq_lock_irq(rq);
}

static DEFINE_PER_CPU(struct balance_callback, core_balance_head);

/*
 * queue_core_balance - 核心 cookie 导致本 CPU 强制空闲时，排入核心均衡回调。
 */
static void queue_core_balance(struct rq *rq)
{
	if (!sched_core_enabled(rq))
		return;

	if (!rq->core->core_cookie)
		return;

	if (!rq->nr_running) /* not forced idle */
		return;

	queue_balance_callback(rq, &per_cpu(core_balance_head, rq->cpu), sched_core_balance);
}

DEFINE_LOCK_GUARD_1(core_lock, int,
		    sched_core_lock(*_T->lock, &_T->flags),
		    sched_core_unlock(*_T->lock, &_T->flags),
		    unsigned long flags)

/*
 * sched_core_cpu_starting - CPU 启动时加入 SMT 核心的共享运行队列状态，
 * 并校验领导队列。
 */
static void sched_core_cpu_starting(unsigned int cpu)
{
	const struct cpumask *smt_mask = cpu_smt_mask(cpu);
	struct rq *rq = cpu_rq(cpu), *core_rq = NULL;
	int t;

	guard(core_lock)(&cpu);

	WARN_ON_ONCE(rq->core != rq);

	/* if we're the first, we'll be our own leader */
	if (cpumask_weight(smt_mask) == 1)
		return;

	/* find the leader */
	for_each_cpu(t, smt_mask) {
		if (t == cpu)
			continue;
		rq = cpu_rq(t);
		if (rq->core == rq) {
			core_rq = rq;
			break;
		}
	}

	if (WARN_ON_ONCE(!core_rq)) /* whoopsie */
		return;

	/* install and validate core_rq */
	for_each_cpu(t, smt_mask) {
		rq = cpu_rq(t);

		if (t == cpu)
			rq->core = core_rq;

		WARN_ON_ONCE(rq->core != core_rq);
	}
}

/*
 * sched_core_cpu_deactivate - 核心领导 CPU 下线时转移共享调度状态，
 * 并重设兄弟队列指针。
 */
static void sched_core_cpu_deactivate(unsigned int cpu)
{
	const struct cpumask *smt_mask = cpu_smt_mask(cpu);
	struct rq *rq = cpu_rq(cpu), *core_rq = NULL;
	int t;

	guard(core_lock)(&cpu);

	/* if we're the last man standing, nothing to do */
	if (cpumask_weight(smt_mask) == 1) {
		WARN_ON_ONCE(rq->core != rq);
		return;
	}

	/* if we're not the leader, nothing to do */
	if (rq->core != rq)
		return;

	/* find a new leader */
	for_each_cpu(t, smt_mask) {
		if (t == cpu)
			continue;
		core_rq = cpu_rq(t);
		break;
	}

	if (WARN_ON_ONCE(!core_rq)) /* impossible */
		return;

	/* copy the shared state to the new leader */
	core_rq->core_task_seq             = rq->core_task_seq;
	core_rq->core_pick_seq             = rq->core_pick_seq;
	core_rq->core_cookie               = rq->core_cookie;
	core_rq->core_forceidle_count      = rq->core_forceidle_count;
	core_rq->core_forceidle_seq        = rq->core_forceidle_seq;
	core_rq->core_forceidle_occupation = rq->core_forceidle_occupation;

	/*
	 * A stale leftover would bias the count forever if this CPU later
	 * returns as its own leader. Move, don't copy.
	 */
	core_rq->core_pick_in_flight       = rq->core_pick_in_flight;
	rq->core_pick_in_flight            = 0;

	/*
	 * Accounting edge for forced idle is handled in pick_next_task().
	 * Don't need another one here, since the hotplug thread shouldn't
	 * have a cookie.
	 */
	core_rq->core_forceidle_start = 0;

	/* install new leader */
	for_each_cpu(t, smt_mask) {
		rq = cpu_rq(t);
		rq->core = core_rq;
	}
}

/*
 * sched_core_cpu_dying - CPU 最终下线时解除其与原核心共享队列的关联。
 */
static inline void sched_core_cpu_dying(unsigned int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	if (rq->core != rq)
		rq->core = rq;
}

#else /* !CONFIG_SCHED_CORE: */

/*
 * sched_core_cpu_starting - 未启用核心调度时的空操作 CPU 启动接口。
 */
static inline void sched_core_cpu_starting(unsigned int cpu) {}
/*
 * sched_core_cpu_deactivate - 未启用核心调度时的空操作 CPU 停用接口。
 */
static inline void sched_core_cpu_deactivate(unsigned int cpu) {}
/*
 * sched_core_cpu_dying - 未启用核心调度时的空操作 CPU 下线接口。
 */
static inline void sched_core_cpu_dying(unsigned int cpu) {}

/*
 * pick_next_task - 未启用核心调度时直接使用通用的下一任务选择逻辑。
 */
static struct task_struct *
pick_next_task(struct rq *rq, struct rq_flags *rf)
{
	return __pick_next_task(rq, rf);
}

#endif /* !CONFIG_SCHED_CORE */

/*
 * Constants for the sched_mode argument of __schedule().
 *
 * The mode argument allows RT enabled kernels to differentiate a
 * preemption from blocking on an 'sleeping' spin/rwlock.
 */
#define SM_IDLE			(-1)
#define SM_NONE			0
#define SM_PREEMPT		1
#define SM_RTLOCK_WAIT		2

/*
 * sched_mode 有两种判法：调试/RCU 将正值视为抢占类切换，实际处理
 * 任务睡眠时却仅把 SM_PREEMPT 当作抢占；RT 锁等待仍需处理阻塞。
 * 这样既保留抢占式 RCU 所需语义，又不丢掉睡眠锁的等待请求。
 */

/*
 * Helper function for __schedule()
 *
 * Tries to deactivate the task, unless the should_block arg
 * is false or if a signal is pending. In the case a signal
 * is pending, marks the task's __state as RUNNING (and clear
 * blocked_on).
 */
/*
 * try_to_block_task - 检查待处理信号和阻塞许可，
 * 决定是否提交任务的睡眠出队处理。
 * 返回 true 表示已处理阻塞请求，实际出队仍可能被调度类延迟。
 */
static bool try_to_block_task(struct rq *rq, struct task_struct *p,
			      unsigned long *task_state_p, bool should_block)
{
	unsigned long task_state = *task_state_p;
	int flags = DEQUEUE_NOCLOCK;

	/*
	 * 最后一次检查信号必须在真正出队之前：可中断等待若已有信号，
	 * 撤销睡眠请求，后续即使仍选中自己也能回到调用者处理信号。
	 * 不可中断状态并非被所有信号唤醒，具体由 signal_pending_state 判定。
	 */
	if (signal_pending_state(task_state, p)) {
		WRITE_ONCE(p->__state, TASK_RUNNING);
		*task_state_p = TASK_RUNNING;
		return false;
	}

	/*
	 * We check should_block after signal_pending because we
	 * will want to wake the task in that case. But if
	 * should_block is false, its likely due to the task being
	 * blocked on a mutex, and we want to keep it on the runqueue
	 * to be selectable for proxy-execution.
	 */
	if (!should_block)
		return false;

	/* D 状态中排除 NOLOAD 和 FROZEN，保存此次阻塞是否计入系统负载。 */
	p->sched_contributes_to_load =
		(task_state & TASK_UNINTERRUPTIBLE) &&
		!(task_state & TASK_NOLOAD) &&
		!(task_state & TASK_FROZEN);

	if (unlikely(is_special_task_state(task_state)))
		flags |= DEQUEUE_SPECIAL;

	/*
	 * __schedule()			ttwu()
	 *   prev_state = prev->state;    if (p->on_rq && ...)
	 *   if (prev_state)		    goto out;
	 *     p->on_rq = 0;		  smp_acquire__after_ctrl_dep();
	 *				  p->state = TASK_WAKING
	 *
	 * Where __schedule() and ttwu() have matching control dependencies.
	 *
	 * After this, schedule() must not care about p->state any more.
	 */
	block_task(rq, p, flags);
	return true;
}

#ifdef CONFIG_SCHED_PROXY_EXEC
/*
 * proxy_resched_idle - 暂时选择空闲任务并保留重调度请求，
 * 使代理执行相关竞态得到推进。
 */
static inline struct task_struct *proxy_resched_idle(struct rq *rq)
{
	put_prev_set_next_task(rq, rq->donor, rq->idle);
	rq_set_donor(rq, rq->idle);
	set_tsk_need_resched(rq->idle);
	return rq->idle;
}

/*
 * __proxy_deactivate - 先撤销对代理资源提供者的当前引用，
 * 再尝试将仍睡眠的提供者阻塞。
 */
static bool __proxy_deactivate(struct rq *rq, struct task_struct *donor)
{
	unsigned long state = READ_ONCE(donor->__state);

	/* Don't deactivate if the state has been changed to TASK_RUNNING */
	if (state == TASK_RUNNING)
		return false;
	/*
	 * Because we got donor from pick_next_task(), it is *crucial*
	 * that we call proxy_resched_idle() before we deactivate it.
	 * As once we deactivate donor, donor->on_rq is set to zero,
	 * which allows ttwu() to immediately try to wake the task on
	 * another rq. So we cannot use *any* references to donor
	 * after that point. So things like cfs_rq->curr or rq->donor
	 * need to be changed from next *before* we deactivate.
	 */
	proxy_resched_idle(rq);
	return try_to_block_task(rq, donor, &state, true);
}

/*
 * proxy_deactivate - 停用无法继续代理执行的提供者，
 * 失败时清除阻塞关系并要求重新选取。
 */
static struct task_struct *proxy_deactivate(struct rq *rq, struct task_struct *donor)
{
	if (!__proxy_deactivate(rq, donor)) {
		/*
		 * XXX: For now, if deactivation failed, set donor
		 * as unblocked, as we aren't doing proxy-migrations
		 * yet (more logic will be needed then).
		 */
		donor->blocked_on = NULL;
	}
	return NULL;
}

/*
 * Find runnable lock owner to proxy for mutex blocked donor
 *
 * Follow the blocked-on relation:
 *   task->blocked_on -> mutex->owner -> task...
 *
 * Lock order:
 *
 *   p->pi_lock
 *     rq->lock
 *       mutex->wait_lock
 *
 * Returns the task that is going to be used as execution context (the one
 * that is actually going to be run on cpu_of(rq)).
 */
/*
 * find_proxy_task - 沿 mutex 阻塞链查找本 CPU 可执行的锁持有者，
 * 代替阻塞任务运行。
 * 返回实际执行任务；关系变化需重选时返回 NULL，
 * 需让出锁推进时返回空闲任务。
 */
static struct task_struct *
find_proxy_task(struct rq *rq, struct task_struct *donor, struct rq_flags *rf)
{
	struct task_struct *owner = NULL;
	int this_cpu = cpu_of(rq);
	struct task_struct *p;
	struct mutex *mutex;

	/*
	 * 代理执行保留 donor 的调度资格，让其等待的锁持有者实际执行，
	 * 从而推进解锁。此实现只追踪本 rq 上可运行的 mutex 拥有者；
	 * 跨 CPU、已阻塞和延迟出队的 owner 走停用/重选分支，不能把
	 * “沿阻塞链找到 owner”直接等同于可以安全切入该任务。
	 *
	 * 例如 A 的调度资格被选中，但 A 等待 B 持有的 mutex：
	 *   最终切入后 donor=A、curr=B，B 消耗 A 的调度资源以推进解锁。
	 * 若 B 又等待 C，则继续沿链寻找可实际执行者。这与直接修改
	 * B->prio 的优先级继承机制不同，不能用执行者的类代替 donor 的类
	 * 做运行记账；也不能把当前实现理解为会自动把远端 owner 迁过来。
	 */
	/* Follow blocked_on chain. */
	for (p = donor; task_is_blocked(p); p = owner) {
		mutex = p->blocked_on;
		/* Something changed in the chain, so pick again */
		if (!mutex)
			return NULL;
		/*
		 * By taking mutex->wait_lock we hold off concurrent mutex_unlock()
		 * and ensure @owner sticks around.
		 *
		 * rq 锁尚不能保护归属未知的 owner；先用 wait_lock 稳住
		 * mutex 的拥有关系，确认 owner 已在本 rq 后，才能依赖 rq 锁
		 * 保证其存活，并继续沿阻塞链查找。
		 */
		guard(raw_spinlock)(&mutex->wait_lock);

		/* Check again that p is blocked with wait_lock held */
		if (mutex != __get_task_blocked_on(p)) {
			/*
			 * Something changed in the blocked_on chain and
			 * we don't know if only at this level. So, let's
			 * just bail out completely and let __schedule()
			 * figure things out (pick_again loop).
			 */
			return NULL;
		}

		owner = __mutex_owner(mutex);
		if (!owner) {
			__clear_task_blocked_on(p, mutex);
			return p;
		}

		if (!READ_ONCE(owner->on_rq) || owner->se.sched_delayed) {
			/* XXX Don't handle blocked owners/delayed dequeue yet */
			return proxy_deactivate(rq, donor);
		}

		if (task_cpu(owner) != this_cpu) {
			/* XXX Don't handle migrations yet */
			return proxy_deactivate(rq, donor);
		}

		if (task_on_rq_migrating(owner)) {
			/*
			 * One of the chain of mutex owners is currently migrating to this
			 * CPU, but has not yet been enqueued because we are holding the
			 * rq lock. As a simple solution, just schedule rq->idle to give
			 * the migration a chance to complete. Much like the migrate_task
			 * case we should end up back in find_proxy_task(), this time
			 * hopefully with all relevant tasks already enqueued.
			 */
			return proxy_resched_idle(rq);
		}

		/*
		 * Its possible to race where after we check owner->on_rq
		 * but before we check (owner_cpu != this_cpu) that the
		 * task on another cpu was migrated back to this cpu. In
		 * that case it could slip by our  checks. So double check
		 * we are still on this cpu and not migrating. If we get
		 * inconsistent results, try again.
		 */
		if (!task_on_rq_queued(owner) || task_cpu(owner) != this_cpu)
			return NULL;

		if (owner == p) {
			/*
			 * 看到自指不一定是死锁：mutex 解锁后的批量唤醒可能正
			 * 卡在本 rq 锁上。先运行 idle 并释放 rq 锁，让唤醒方
			 * 完成状态更新，再重新选取，不能持锁等待它完成。
			 *
			 * It's possible we interleave with mutex_unlock like:
			 *
			 *				lock(&rq->lock);
			 *				  find_proxy_task()
			 * mutex_unlock()
			 *   lock(&wait_lock);
			 *   donor(owner) = current->blocked_donor;
			 *   unlock(&wait_lock);
			 *
			 *   wake_up_q();
			 *     ...
			 *       ttwu_runnable()
			 *         __task_rq_lock()
			 *				  lock(&wait_lock);
			 *				  owner == p
			 *
			 * Which leaves us to finish the ttwu_runnable() and make it go.
			 *
			 * So schedule rq->idle so that ttwu_runnable() can get the rq
			 * lock and mark owner as running.
			 */
			return proxy_resched_idle(rq);
		}
		/*
		 * OK, now we're absolutely sure @owner is on this
		 * rq, therefore holding @rq->lock is sufficient to
		 * guarantee its existence, as per ttwu_remote().
		 */
	}

	WARN_ON_ONCE(owner && !owner->on_rq);
	return owner;
}
#else /* SCHED_PROXY_EXEC */
/*
 * find_proxy_task - 未编译代理执行时报告意外调用，
 * 并返回原调度资源提供者。
 */
static struct task_struct *
find_proxy_task(struct rq *rq, struct task_struct *donor, struct rq_flags *rf)
{
	WARN_ONCE(1, "This should never be called in the !SCHED_PROXY_EXEC case\n");
	return donor;
}
#endif /* SCHED_PROXY_EXEC */

/*
 * __schedule() is the main scheduler function.
 *
 * The main means of driving the scheduler and thus entering this function are:
 *
 *   1. Explicit blocking: mutex, semaphore, waitqueue, etc.
 *
 *   2. TIF_NEED_RESCHED flag is checked on interrupt and userspace return
 *      paths. For example, see arch/x86/entry_64.S.
 *
 *      To drive preemption between tasks, the scheduler sets the flag in timer
 *      interrupt handler sched_tick().
 *
 *   3. Wakeups don't really cause entry into schedule(). They add a
 *      task to the run-queue and that's it.
 *
 *      Now, if the new task added to the run-queue preempts the current
 *      task, then the wakeup sets TIF_NEED_RESCHED and schedule() gets
 *      called on the nearest possible occasion:
 *
 *       - If the kernel is preemptible (CONFIG_PREEMPTION=y):
 *
 *         - in syscall or exception context, at the next outmost
 *           preempt_enable(). (this might be as soon as the wake_up()'s
 *           spin_unlock()!)
 *
 *         - in IRQ context, return from interrupt-handler to
 *           preemptible context
 *
 *       - If the kernel is not preemptible (CONFIG_PREEMPTION is not set)
 *         then at the next:
 *
 *          - cond_resched() call
 *          - explicit schedule() call
 *          - return from syscall or exception to user-space
 *          - return from interrupt-handler to user-space
 *
 * WARNING: must be called with preemption disabled!
 */
/*
 * __schedule - 调度器主流程：处理阻塞、选择调度资源及代理任务，
 * 并执行上下文切换。
 * 调用时必须禁止抢占；sched_mode 区分空闲、普通、抢占及 RT 锁等待调度。
 */
static void __sched notrace __schedule(int sched_mode)
{
	struct task_struct *prev, *next;
	/*
	 * On PREEMPT_RT kernel, SM_RTLOCK_WAIT is noted
	 * as a preemption by schedule_debug() and RCU.
	 */
	bool preempt = sched_mode > SM_NONE;
	bool is_switch = false;
	unsigned long *switch_count;
	unsigned long prev_state;
	struct rq_flags rf;
	struct rq *rq;
	int cpu;

	/*
	 * 按六步阅读下面的主体：
	 *   固定本 CPU 并关闭中断 -> 锁住 rq、更新时钟 -> 处理 prev 的阻塞
	 *   -> 调度类选 donor（按需找代理执行者）-> 发布 curr/统计
	 *   -> 换栈并由切入者解锁，或原地解锁继续执行。
	 * 只有最后选出的 next != prev 才发生真正上下文切换。
	 */
	/* Trace preemptions consistently with task switches */
	trace_sched_entry_tp(sched_mode == SM_PREEMPT);

	cpu = smp_processor_id();
	rq = cpu_rq(cpu);
	prev = rq->curr;

	schedule_debug(prev, preempt);

	if (sched_feat(HRTICK) || sched_feat(HRTICK_DL))
		hrtick_clear(rq);

	klp_sched_try_switch(prev);

	local_irq_disable();
	rcu_note_context_switch(preempt);

	/*
	 * Make sure that signal_pending_state()->signal_pending() below
	 * can't be reordered with __set_current_state(TASK_INTERRUPTIBLE)
	 * done by the caller to avoid the race with signal_wake_up():
	 *
	 * __set_current_state(@state)		signal_wake_up()
	 * schedule()				  set_tsk_thread_flag(p, TIF_SIGPENDING)
	 *					  wake_up_state(p, state)
	 *   LOCK rq->lock			    LOCK p->pi_state
	 *   smp_mb__after_spinlock()		    smp_mb__after_spinlock()
	 *     if (signal_pending_state())	    if (p->state & @state)
	 *
	 * Also, the membarrier system call requires a full memory barrier
	 * after coming from user-space, before storing to rq->curr; this
	 * barrier matches a full barrier in the proximity of the membarrier
	 * system call exit.
	 *
	 * 加锁后的完整屏障有两项职责：让睡眠状态先于待处理信号检查
	 * 可见，防止漏掉信号唤醒；满足 membarrier 对发布 rq->curr 前
	 * 访存顺序的要求。仅靠自旋锁的 acquire 语义不能替代它。
	 */
	rq_lock(rq, &rf);
	smp_mb__after_spinlock();

	/*
	 * Promote REQ to ACT
	 * 将此前请求的 RQCF_REQ_SKIP 提升为 RQCF_ACT_SKIP，避免紧接着
	 * 重复更新时钟；随后清除跳过标志，保留本轮时钟已就绪的标记。
	 */
	rq->clock_update_flags <<= 1;
	update_rq_clock(rq);
	rq->clock_update_flags = RQCF_UPDATED;

	/*
	 * 默认记入非自愿切换 nivcsw；若下面处理主动阻塞请求，则改用
	 * nvcsw。这里只选择计数器，实际选出不同任务后才递增。
	 */
	switch_count = &prev->nivcsw;

	/* Task state changes only considers SM_PREEMPT as preemption */
	preempt = sched_mode == SM_PREEMPT;

	/*
	 * We must load prev->state once (task_struct::state is volatile), such
	 * that we form a control dependency vs deactivate_task() below.
	 *
	 * 请求睡眠与真正调用 schedule() 之间也可能发生抢占。因此
	 * SM_PREEMPT 只暂停执行，不依据睡眠状态替任务提交阻塞；
	 * 普通调度及 RT 锁等待才在下面检查并处理阻塞请求。
	 */
	prev_state = READ_ONCE(prev->__state);
	if (sched_mode == SM_IDLE) {
		/* SCX must consult the BPF scheduler to tell if rq is empty */
		if (!rq->nr_running && !scx_enabled()) {
			next = prev;
			goto picked;
		}
	} else if (!preempt && prev_state) {
		/*
		 * We pass task_is_blocked() as the should_block arg
		 * in order to keep mutex-blocked tasks on the runqueue
		 * for slection with proxy-exec (without proxy-exec
		 * task_is_blocked() will always be false).
		 */
		try_to_block_task(rq, prev, &prev_state,
				  !task_is_blocked(prev));
		switch_count = &prev->nvcsw;
	}

pick_again:
	assert_balance_callbacks_empty(rq);
	next = pick_next_task(rq, &rf);
	/*
	 * donor 是调度类选中的资源提供者，决定调度优先级与运行记账；
	 * 代理执行时，它可能正等待 mutex，而真正执行的 next 是锁持有者。
	 * 因此 rq->donor 与稍后写入的 rq->curr 不一定是同一个任务。
	 */
	rq_set_donor(rq, next);
	if (unlikely(task_is_blocked(next))) {
		next = find_proxy_task(rq, next, &rf);
		if (!next)
			goto pick_again;
		/* 临时运行 idle 以推进竞态，保留重调度请求以便重试。 */
		if (next == rq->idle)
			goto keep_resched;
	}
picked:
	clear_tsk_need_resched(prev);
	clear_preempt_need_resched();
keep_resched:
	rq->last_seen_need_resched_ns = 0;

	is_switch = prev != next;
	if (likely(is_switch)) {
		rq->nr_switches++;
		/*
		 * RCU users of rcu_dereference(rq->curr) may not see
		 * changes to task_struct made by pick_next_task().
		 */
		RCU_INIT_POINTER(rq->curr, next);

		/*
		 * The membarrier system call requires each architecture
		 * to have a full memory barrier after updating
		 * rq->curr, before returning to user-space.
		 *
		 * Here are the schemes providing that barrier on the
		 * various architectures:
		 * - mm ? switch_mm() : mmdrop() for x86, s390, sparc, PowerPC,
		 *   RISC-V.  switch_mm() relies on membarrier_arch_switch_mm()
		 *   on PowerPC and on RISC-V.
		 * - finish_lock_switch() for weakly-ordered
		 *   architectures where spin_unlock is a full barrier,
		 * - switch_to() for arm64 (weakly-ordered, spin_unlock
		 *   is a RELEASE barrier),
		 *
		 * The barrier matches a full barrier in the proximity of
		 * the membarrier system call entry.
		 *
		 * On RISC-V, this barrier pairing is also needed for the
		 * SYNC_CORE command when switching between processes, cf.
		 * the inline comments in membarrier_arch_switch_mm().
		 */
		++*switch_count;

		migrate_disable_switch(rq, prev);
		psi_account_irqtime(rq, prev, next);
		psi_sched_switch(prev, next, !task_on_rq_queued(prev) ||
					     prev->se.sched_delayed);

		trace_sched_switch(preempt, prev, next, prev_state);

		/* Also unlocks the rq: */
		rq = context_switch(rq, prev, next, &rf);
	} else {
		/*
		 * 仍选中原任务时无需换栈，也不会进入 finish_task_switch()；
		 * 因而在这里自行执行均衡回调并释放 rq 锁、重新开启中断。
		 */
		rq_unpin_lock(rq, &rf);
		__balance_callbacks(rq);
		raw_spin_rq_unlock_irq(rq);
	}
	trace_sched_exit_tp(is_switch);
}

/*
 * do_task_dead - 将当前任务置为死亡状态并进行最后一次调度，
 * 此函数永不返回。
 */
void __noreturn do_task_dead(void)
{
	/* Causes final put_task_struct in finish_task_switch(): */
	set_special_state(TASK_DEAD);

	/* Tell freezer to ignore us: */
	current->flags |= PF_NOFREEZE;

	__schedule(SM_NONE);
	BUG();

	/* Avoid "noreturn function does return" - but don't continue if BUG() is a NOP: */
	for (;;)
		cpu_relax();
}

/*
 * sched_submit_work - 睡眠前通知工作线程管理器并提交积压的块 I/O，
 * 避免并发不足和死锁。
 */
static inline void sched_submit_work(struct task_struct *tsk)
{
	static DEFINE_WAIT_OVERRIDE_MAP(sched_map, LD_WAIT_CONFIG);
	unsigned int task_flags;

	/*
	 * Establish LD_WAIT_CONFIG context to ensure none of the code called
	 * will use a blocking primitive -- which would lead to recursion.
	 */
	lock_map_acquire_try(&sched_map);

	task_flags = tsk->flags;
	/*
	 * If a worker goes to sleep, notify and ask workqueue whether it
	 * wants to wake up a task to maintain concurrency.
	 */
	if (task_flags & PF_WQ_WORKER)
		wq_worker_sleeping(tsk);
	else if (task_flags & PF_IO_WORKER)
		io_wq_worker_sleeping(tsk);

	/*
	 * spinlock and rwlock must not flush block requests.  This will
	 * deadlock if the callback attempts to acquire a lock which is
	 * already acquired.
	 */
	WARN_ON_ONCE(current->__state & TASK_RTLOCK_WAIT);

	/*
	 * If we are going to sleep and we have plugged IO queued,
	 * make sure to submit it to avoid deadlocks.
	 *
	 * 典型依赖环是：当前任务等 I/O 完成，但请求还留在自己的 plug
	 * 批次里尚未提交。睡眠前须先提交，让完成事件能产生。
	 * 抢占仅暂停任务，不经过这套主动睡眠的准备流程。
	 */
	blk_flush_plug(tsk->plug, true);

	lock_map_release(&sched_map);
}

/*
 * sched_update_worker - 任务恢复运行后通知 workqueue 或 io-wq，
 * 更新工作线程运行状态。
 */
static void sched_update_worker(struct task_struct *tsk)
{
	if (tsk->flags & (PF_WQ_WORKER | PF_IO_WORKER)) {
		if (tsk->flags & PF_WQ_WORKER)
			wq_worker_running(tsk);
		else
			io_wq_worker_running(tsk);
	}
}

/*
 * __schedule_loop - 在禁止抢占的窗口内反复调度，
 * 直到不再有待处理重调度请求。
 */
static __always_inline void __schedule_loop(int sched_mode)
{
	do {
		preempt_disable();
		__schedule(sched_mode);
		/*
		 * 恢复抢占计数但不在这里递归触发调度。若切换期间又产生了
		 * 重调度请求，则由循环处理，避免遗漏或递归嵌套。
		 */
		sched_preempt_enable_no_resched();
	} while (need_resched());
}

/*
 * schedule - 通用主动调度入口，
 * 执行睡眠前工作、调度循环及工作线程恢复通知。
 */
asmlinkage __visible void __sched schedule(void)
{
	struct task_struct *tsk = current;

#ifdef CONFIG_RT_MUTEXES
	lockdep_assert(!tsk->sched_rt_mutex);
#endif

	/*
	 * 调用者负责事先设置睡眠状态并检查等待条件。若仍是 TASK_RUNNING，
	 * schedule() 只是一次重新选择任务的机会，不会主动把当前任务阻塞。
	 */
	if (!task_is_running(tsk))
		sched_submit_work(tsk);
	__schedule_loop(SM_NONE);
	sched_update_worker(tsk);
}
EXPORT_SYMBOL(schedule);

/*
 * synchronize_rcu_tasks() makes sure that no task is stuck in preempted
 * state (have scheduled out non-voluntarily) by making sure that all
 * tasks have either left the run queue or have gone into user space.
 * As idle tasks do not do either, they must not ever be preempted
 * (schedule out non-voluntarily).
 *
 * schedule_idle() is similar to schedule_preempt_disable() except that it
 * never enables preemption because it does not call sched_submit_work().
 */
/*
 * schedule_idle - 空闲任务专用调度入口，
 * 整个调用保持禁止抢占，不执行睡眠前工作。
 */
void __sched schedule_idle(void)
{
	/*
	 * As this skips calling sched_submit_work(), which the idle task does
	 * regardless because that function is a NOP when the task is in a
	 * TASK_RUNNING state, make sure this isn't used someplace that the
	 * current task can be in any other state. Note, idle is always in the
	 * TASK_RUNNING state.
	 */
	WARN_ON_ONCE(current->__state);
	do {
		__schedule(SM_IDLE);
	} while (need_resched());
}

#if defined(CONFIG_CONTEXT_TRACKING_USER) && !defined(CONFIG_HAVE_CONTEXT_TRACKING_USER_OFFSTACK)
/*
 * schedule_user - 从用户上下文跟踪状态进入调度，
 * 并在返回后恢复原上下文状态。
 */
asmlinkage __visible void __sched schedule_user(void)
{
	/*
	 * If we come here after a random call to set_need_resched(),
	 * or we have been woken up remotely but the IPI has not yet arrived,
	 * we haven't yet exited the RCU idle mode. Do it here manually until
	 * we find a better solution.
	 *
	 * NB: There are buggy callers of this function.  Ideally we
	 * should warn if prev_state != CT_STATE_USER, but that will trigger
	 * too frequently to make sense yet.
	 */
	enum ctx_state prev_state = exception_enter();
	schedule();
	exception_exit(prev_state);
}
#endif

/**
 * schedule_preempt_disabled - called with preemption disabled
 * 在抢占计数为 1 时临时允许调度，返回前重新禁止抢占。
 *
 * Returns with preemption disabled. Note: preempt_count must be 1
 */
void __sched schedule_preempt_disabled(void)
{
	sched_preempt_enable_no_resched();
	schedule();
	preempt_disable();
}

#ifdef CONFIG_PREEMPT_RT
/*
 * schedule_rtlock - 为 PREEMPT_RT 的睡眠式自旋锁等待执行专用调度循环。
 */
void __sched notrace schedule_rtlock(void)
{
	__schedule_loop(SM_RTLOCK_WAIT);
}
NOKPROBE_SYMBOL(schedule_rtlock);
#endif

/*
 * preempt_schedule_common - 执行带抢占延迟跟踪的抢占调度循环，
 * 避免函数跟踪递归。
 */
static void __sched notrace preempt_schedule_common(void)
{
	do {
		/*
		 * Because the function tracer can trace preempt_count_sub()
		 * and it also uses preempt_enable/disable_notrace(), if
		 * NEED_RESCHED is set, the preempt_enable_notrace() called
		 * by the function tracer will call this function again and
		 * cause infinite recursion.
		 *
		 * Preemption must be disabled here before the function
		 * tracer can trace. Break up preempt_disable() into two
		 * calls. One to disable preemption without fear of being
		 * traced. The other to still record the preemption latency,
		 * which can also be traced by the function tracer.
		 */
		preempt_disable_notrace();
		preempt_latency_start(1);
		__schedule(SM_PREEMPT);
		preempt_latency_stop(1);
		preempt_enable_no_resched_notrace();

		/*
		 * Check again in case we missed a preemption opportunity
		 * between schedule and now.
		 */
	} while (need_resched());
}

#ifdef CONFIG_PREEMPTION
/*
 * This is the entry point to schedule() from in-kernel preemption
 * off of preempt_enable.
 */
/*
 * preempt_schedule - 内核恢复抢占时的调度入口，
 * 仅在当前上下文允许抢占时进行调度。
 */
asmlinkage __visible void __sched notrace preempt_schedule(void)
{
	/*
	 * If there is a non-zero preempt_count or interrupts are disabled,
	 * we do not want to preempt the current task. Just return..
	 *
	 * need_resched 表示“应该换人”，preemptible 表示“此刻允许换人”。
	 * 例如持有 raw 自旋锁时请求可以积累，必须等最外层临界区结束
	 * 才能处理；本入口不会把调用者设置的睡眠状态提交成一次阻塞。
	 */
	if (likely(!preemptible()))
		return;
	preempt_schedule_common();
}
NOKPROBE_SYMBOL(preempt_schedule);
EXPORT_SYMBOL(preempt_schedule);

#ifdef CONFIG_PREEMPT_DYNAMIC
# ifdef CONFIG_HAVE_PREEMPT_DYNAMIC_CALL
#  ifndef preempt_schedule_dynamic_enabled
#   define preempt_schedule_dynamic_enabled	preempt_schedule
#   define preempt_schedule_dynamic_disabled	NULL
#  endif
DEFINE_STATIC_CALL(preempt_schedule, preempt_schedule_dynamic_enabled);
EXPORT_STATIC_CALL_TRAMP(preempt_schedule);
# elif defined(CONFIG_HAVE_PREEMPT_DYNAMIC_KEY)
static DEFINE_STATIC_KEY_TRUE(sk_dynamic_preempt_schedule);
/*
 * dynamic_preempt_schedule - 根据动态静态分支决定是否进入普通内核抢占调度。
 */
void __sched notrace dynamic_preempt_schedule(void)
{
	if (!static_branch_unlikely(&sk_dynamic_preempt_schedule))
		return;
	preempt_schedule();
}
NOKPROBE_SYMBOL(dynamic_preempt_schedule);
EXPORT_SYMBOL(dynamic_preempt_schedule);
# endif
#endif /* CONFIG_PREEMPT_DYNAMIC */

/**
 * preempt_schedule_notrace - preempt_schedule called by tracing
 * 为跟踪代码提供无递归抢占入口，必要时先退出用户上下文再进行调度。
 *
 * The tracing infrastructure uses preempt_enable_notrace to prevent
 * recursion and tracing preempt enabling caused by the tracing
 * infrastructure itself. But as tracing can happen in areas coming
 * from userspace or just about to enter userspace, a preempt enable
 * can occur before user_exit() is called. This will cause the scheduler
 * to be called when the system is still in usermode.
 *
 * To prevent this, the preempt_enable_notrace will use this function
 * instead of preempt_schedule() to exit user context if needed before
 * calling the scheduler.
 */
asmlinkage __visible void __sched notrace preempt_schedule_notrace(void)
{
	enum ctx_state prev_ctx;

	if (likely(!preemptible()))
		return;

	do {
		/*
		 * Because the function tracer can trace preempt_count_sub()
		 * and it also uses preempt_enable/disable_notrace(), if
		 * NEED_RESCHED is set, the preempt_enable_notrace() called
		 * by the function tracer will call this function again and
		 * cause infinite recursion.
		 *
		 * Preemption must be disabled here before the function
		 * tracer can trace. Break up preempt_disable() into two
		 * calls. One to disable preemption without fear of being
		 * traced. The other to still record the preemption latency,
		 * which can also be traced by the function tracer.
		 */
		preempt_disable_notrace();
		preempt_latency_start(1);
		/*
		 * Needs preempt disabled in case user_exit() is traced
		 * and the tracer calls preempt_enable_notrace() causing
		 * an infinite recursion.
		 */
		prev_ctx = exception_enter();
		__schedule(SM_PREEMPT);
		exception_exit(prev_ctx);

		preempt_latency_stop(1);
		preempt_enable_no_resched_notrace();
	} while (need_resched());
}
EXPORT_SYMBOL_GPL(preempt_schedule_notrace);

#ifdef CONFIG_PREEMPT_DYNAMIC
# if defined(CONFIG_HAVE_PREEMPT_DYNAMIC_CALL)
#  ifndef preempt_schedule_notrace_dynamic_enabled
#   define preempt_schedule_notrace_dynamic_enabled	preempt_schedule_notrace
#   define preempt_schedule_notrace_dynamic_disabled	NULL
#  endif
DEFINE_STATIC_CALL(preempt_schedule_notrace, preempt_schedule_notrace_dynamic_enabled);
EXPORT_STATIC_CALL_TRAMP(preempt_schedule_notrace);
# elif defined(CONFIG_HAVE_PREEMPT_DYNAMIC_KEY)
static DEFINE_STATIC_KEY_TRUE(sk_dynamic_preempt_schedule_notrace);
/*
 * dynamic_preempt_schedule_notrace - 动态控制无递归抢占调度入口。
 * 此入口供跟踪代码使用。
 */
void __sched notrace dynamic_preempt_schedule_notrace(void)
{
	if (!static_branch_unlikely(&sk_dynamic_preempt_schedule_notrace))
		return;
	preempt_schedule_notrace();
}
NOKPROBE_SYMBOL(dynamic_preempt_schedule_notrace);
EXPORT_SYMBOL(dynamic_preempt_schedule_notrace);
# endif
#endif

#endif /* CONFIG_PREEMPTION */

/*
 * This is the entry point to schedule() from kernel preemption
 * off of IRQ context.
 * Note, that this is called and return with IRQs disabled. This will
 * protect us against recursive calling from IRQ contexts.
 */
/*
 * preempt_schedule_irq - 中断返回路径的抢占入口，
 * 调用和返回时均保持中断关闭。
 * 要求进入时抢占计数为 0，调度过程中临时开启中断。
 */
asmlinkage __visible void __sched preempt_schedule_irq(void)
{
	enum ctx_state prev_state;

	/* Catch callers which need to be fixed */
	BUG_ON(preempt_count() || !irqs_disabled());

	prev_state = exception_enter();

	do {
		preempt_disable();
		/*
		 * 禁止抢占以防递归，允许调度前的钩子处理中断；
		 * __schedule 内部会重新关中断。返回后先关中断，再恢复计数，
		 * 保持 IRQ 返回入口约定，最后检查期间新产生的重调度请求。
		 */
		local_irq_enable();
		__schedule(SM_PREEMPT);
		local_irq_disable();
		sched_preempt_enable_no_resched();
	} while (need_resched());

	exception_exit(prev_state);
}

/*
 * default_wake_function - 等待队列的默认唤醒回调，
 * 按给定模式唤醒条目关联的任务。
 */
int default_wake_function(wait_queue_entry_t *curr, unsigned mode, int wake_flags,
			  void *key)
{
	WARN_ON_ONCE(wake_flags & ~(WF_SYNC|WF_CURRENT_CPU));
	return try_to_wake_up(curr->private, mode, wake_flags);
}
EXPORT_SYMBOL(default_wake_function);

/*
 * __setscheduler_class - 根据有效优先级和调度策略，
 * 选择截止时间、实时、扩展或公平调度类。
 */
const struct sched_class *__setscheduler_class(int policy, int prio)
{
	if (dl_prio(prio))
		return &dl_sched_class;

	if (rt_prio(prio))
		return &rt_sched_class;

#ifdef CONFIG_SCHED_CLASS_EXT
	if (task_should_scx(policy))
		return &ext_sched_class;
#endif

	return &fair_sched_class;
}

#ifdef CONFIG_RT_MUTEXES

/*
 * Would be more useful with typeof()/auto_type but they don't mix with
 * bit-fields. Since it's a local thing, use int. Keep the generic sounding
 * name such that if someone were to implement this function we get to compare
 * notes.
 */
#define fetch_and_set(x, v) ({ int _x = (x); (x) = (v); _x; })

/*
 * rt_mutex_pre_schedule - 标记进入 RT mutex 调度区间，
 * 并完成任务睡眠前的工作提交。
 */
void rt_mutex_pre_schedule(void)
{
	lockdep_assert(!fetch_and_set(current->sched_rt_mutex, 1));
	sched_submit_work(current);
}

/*
 * Used within the futex syscall context, skips sched_submit_work() because none
 * its work will be done. Asserts ensure that it is indeed the case.
 */
/*
 * rt_mutex_futex_pre_schedule - 为 futex 的 RT mutex 等待建立调度标记，
 * 跳过无需执行的提交。
 */
void rt_mutex_futex_pre_schedule(void)
{
	lockdep_assert(!(current->flags & (PF_WQ_WORKER | PF_IO_WORKER)));
	lockdep_assert(!current->plug);
	lockdep_assert(!fetch_and_set(current->sched_rt_mutex, 1));
}

/*
 * rt_mutex_schedule - 在已建立 RT mutex 调度标记的区间内
 * 执行普通阻塞调度循环。
 */
void rt_mutex_schedule(void)
{
	lockdep_assert(current->sched_rt_mutex);
	__schedule_loop(SM_NONE);
}

/*
 * rt_mutex_post_schedule - RT mutex 等待结束后恢复工作线程状态并清除调度标记。
 */
void rt_mutex_post_schedule(void)
{
	sched_update_worker(current);
	lockdep_assert(fetch_and_set(current->sched_rt_mutex, 0));
}

/*
 * rt_mutex_futex_post_schedule - futex 的 RT mutex 等待结束后清除调度标记。
 */
void rt_mutex_futex_post_schedule(void)
{
	lockdep_assert(fetch_and_set(current->sched_rt_mutex, 0));
}

/*
 * rt_mutex_setprio - set the current priority of a task
 * @p: task to boost
 * @pi_task: donor task
 *
 * This function changes the 'effective' priority of a task. It does
 * not touch ->normal_prio like __setscheduler().
 *
 * Used by the rt_mutex code to implement priority inheritance
 * logic. Call site only calls if the priority of the task changed.
 */
/*
 * rt_mutex_setprio - 根据优先级继承提供者更新任务有效优先级及调度类，
 * 不改变正常优先级。
 * 调用者须持有 pi_lock；函数协调队列重排、截止时间继承和均衡回调。
 */
void rt_mutex_setprio(struct task_struct *p, struct task_struct *pi_task)
{
	int prio, oldprio, queue_flag =
		DEQUEUE_SAVE | DEQUEUE_MOVE | DEQUEUE_NOCLOCK;
	const struct sched_class *prev_class, *next_class;
	struct rq_flags rf;
	struct rq *rq;

	/* XXX used to be waiter->prio, not waiter->task->prio */
	/*
	 * normal_prio 保留任务本来应有的优先级，prio 是当前有效值。
	 * 提升可让普通任务临时进入 RT/DL 类；解除继承时再依据本来策略
	 * 恢复，所以这里不能覆盖 policy 或 normal_prio。
	 */
	prio = __rt_effective_prio(pi_task, p->normal_prio);

	/*
	 * If nothing changed; bail early.
	 */
	if (p->pi_top_task == pi_task && prio == p->prio && !dl_prio(prio))
		return;

	rq = __task_rq_lock(p, &rf);
	update_rq_clock(rq);
	/*
	 * Set under pi_lock && rq->lock, such that the value can be used under
	 * either lock.
	 *
	 * Note that there is loads of tricky to make this pointer cache work
	 * right. rt_mutex_slowunlock()+rt_mutex_postunlock() work together to
	 * ensure a task is de-boosted (pi_task is set to NULL) before the
	 * task is allowed to run again (and can exit). This ensures the pointer
	 * points to a blocked task -- which guarantees the task is present.
	 */
	p->pi_top_task = pi_task;

	/*
	 * For FIFO/RR we only need to set prio, if that matches we're done.
	 */
	if (prio == p->prio && !dl_prio(prio))
		goto out_unlock;

	/*
	 * Idle task boosting is a no-no in general. There is one
	 * exception, when PREEMPT_RT and NOHZ is active:
	 *
	 * The idle task calls get_next_timer_interrupt() and holds
	 * the timer wheel base->lock on the CPU and another CPU wants
	 * to access the timer (probably to cancel it). We can safely
	 * ignore the boosting request, as the idle CPU runs this code
	 * with interrupts disabled and will complete the lock
	 * protected section without being interrupted. So there is no
	 * real need to boost.
	 */
	if (unlikely(p == rq->idle)) {
		WARN_ON(p != rq->curr);
		WARN_ON(p->pi_blocked_on);
		goto out_unlock;
	}

	trace_sched_pi_setprio(p, pi_task);
	oldprio = p->prio;

	if (oldprio == prio && !dl_prio(prio))
		queue_flag &= ~DEQUEUE_MOVE;

	prev_class = p->sched_class;
	next_class = __setscheduler_class(p->policy, prio);

	if (prev_class != next_class && p->se.sched_delayed)
		dequeue_task(rq, p, DEQUEUE_SLEEP | DEQUEUE_DELAYED | DEQUEUE_NOCLOCK);

	/*
	 * sched_change 的作用域入口保存并摘下任务，出口按新类/新优先级
	 * 恢复队列与当前实体；类切换通知在队列已恢复后完成。
	 */
	scoped_guard (sched_change, p, queue_flag) {
		/*
		 * Boosting condition are:
		 * 1. -rt task is running and holds mutex A
		 *      --> -dl task blocks on mutex A
		 *
		 * 2. -dl task is running and holds mutex A
		 *      --> -dl task blocks on mutex A and could preempt the
		 *          running task
		 */
		if (dl_prio(prio)) {
			/* DL 还需关联提供者的截止时间实体。 */
			if (!dl_prio(p->normal_prio) ||
			    (pi_task && dl_prio(pi_task->prio) &&
			     dl_entity_preempt(&pi_task->dl, &p->dl))) {
				p->dl.pi_se = pi_task->dl.pi_se;
				scope->flags |= ENQUEUE_REPLENISH;
			} else {
				p->dl.pi_se = &p->dl;
			}
		} else if (rt_prio(prio)) {
			if (dl_prio(oldprio))
				p->dl.pi_se = &p->dl;
			if (oldprio < prio)
				scope->flags |= ENQUEUE_HEAD;
		} else {
			if (dl_prio(oldprio))
				p->dl.pi_se = &p->dl;
			if (rt_prio(oldprio))
				p->rt.timeout = 0;
		}

		p->sched_class = next_class;
		p->prio = prio;

		check_class_changing(rq, p, prev_class);
	}

	check_class_changed(rq, p, prev_class, oldprio);
out_unlock:
	/* Avoid rq from going away on us: */
	preempt_disable();

	rq_unpin_lock(rq, &rf);
	__balance_callbacks(rq);
	raw_spin_rq_unlock(rq);

	preempt_enable();
}
#endif /* CONFIG_RT_MUTEXES */

#if !defined(CONFIG_PREEMPTION) || defined(CONFIG_PREEMPT_DYNAMIC)
/*
 * __cond_resched - 上下文允许且存在重调度请求时主动让出 CPU，
 * 必要时报告 RCU 静默状态。
 * 实际发生调度返回 1，否则返回 0。
 */
int __sched __cond_resched(void)
{
	if (should_resched(0) && !irqs_disabled()) {
		preempt_schedule_common();
		return 1;
	}
	/*
	 * In PREEMPT_RCU kernels, ->rcu_read_lock_nesting tells the tick
	 * whether the current CPU is in an RCU read-side critical section,
	 * so the tick can report quiescent states even for CPUs looping
	 * in kernel context.  In contrast, in non-preemptible kernels,
	 * RCU readers leave no in-memory hints, which means that CPU-bound
	 * processes executing in kernel context might never report an
	 * RCU quiescent state.  Therefore, the following code causes
	 * cond_resched() to report a quiescent state, but only when RCU
	 * is in urgent need of one.
	 * A third case, preemptible, but non-PREEMPT_RCU provides for
	 * urgently needed quiescent states via rcu_flavor_sched_clock_irq().
	 */
#ifndef CONFIG_PREEMPT_RCU
	rcu_all_qs();
#endif
	return 0;
}
EXPORT_SYMBOL(__cond_resched);
#endif

#ifdef CONFIG_PREEMPT_DYNAMIC
# ifdef CONFIG_HAVE_PREEMPT_DYNAMIC_CALL
#  define cond_resched_dynamic_enabled	__cond_resched
#  define cond_resched_dynamic_disabled	((void *)&__static_call_return0)
DEFINE_STATIC_CALL_RET0(cond_resched, __cond_resched);
EXPORT_STATIC_CALL_TRAMP(cond_resched);

#  define might_resched_dynamic_enabled	__cond_resched
#  define might_resched_dynamic_disabled ((void *)&__static_call_return0)
DEFINE_STATIC_CALL_RET0(might_resched, __cond_resched);
EXPORT_STATIC_CALL_TRAMP(might_resched);
# elif defined(CONFIG_HAVE_PREEMPT_DYNAMIC_KEY)
static DEFINE_STATIC_KEY_FALSE(sk_dynamic_cond_resched);
/*
 * dynamic_cond_resched - 动态抢占模式允许时执行条件调度，否则直接返回 0。
 */
int __sched dynamic_cond_resched(void)
{
	if (!static_branch_unlikely(&sk_dynamic_cond_resched))
		return 0;
	return __cond_resched();
}
EXPORT_SYMBOL(dynamic_cond_resched);

static DEFINE_STATIC_KEY_FALSE(sk_dynamic_might_resched);
/*
 * dynamic_might_resched - 按动态开关执行自愿抢占点的条件调度。
 */
int __sched dynamic_might_resched(void)
{
	if (!static_branch_unlikely(&sk_dynamic_might_resched))
		return 0;
	return __cond_resched();
}
EXPORT_SYMBOL(dynamic_might_resched);
# endif
#endif /* CONFIG_PREEMPT_DYNAMIC */

/*
 * __cond_resched_lock() - if a reschedule is pending, drop the given lock,
 * call schedule, and on return reacquire the lock.
 *
 * This works OK both with and without CONFIG_PREEMPTION. We do strange low-level
 * operations here to prevent schedule() from being called twice (once via
 * spin_unlock(), once by hand).
 */
/*
 * __cond_resched_lock - 需要调度或锁竞争让步时释放自旋锁，
 * 尝试调度后重新加锁。
 * 返回 1 表示经历了解锁让步，返回 0 表示保持原锁状态未让步。
 */
int __cond_resched_lock(spinlock_t *lock)
{
	int resched = should_resched(PREEMPT_LOCK_OFFSET);
	int ret = 0;

	lockdep_assert_held(lock);

	if (spin_needbreak(lock) || resched) {
		/*
		 * 返回时虽重新持锁，但临界区已被打断，共享状态可能变化；
		 * 调用者不能保留仅由该锁保证有效的遍历位置或条件假设。
		 */
		spin_unlock(lock);
		if (!_cond_resched())
			cpu_relax();
		ret = 1;
		spin_lock(lock);
	}
	return ret;
}
EXPORT_SYMBOL(__cond_resched_lock);

/*
 * __cond_resched_rwlock_read - 必要时释放读锁并尝试条件调度，
 * 返回前重新获取读锁。
 */
int __cond_resched_rwlock_read(rwlock_t *lock)
{
	int resched = should_resched(PREEMPT_LOCK_OFFSET);
	int ret = 0;

	lockdep_assert_held_read(lock);

	if (rwlock_needbreak(lock) || resched) {
		read_unlock(lock);
		if (!_cond_resched())
			cpu_relax();
		ret = 1;
		read_lock(lock);
	}
	return ret;
}
EXPORT_SYMBOL(__cond_resched_rwlock_read);

/*
 * __cond_resched_rwlock_write - 必要时释放写锁并尝试条件调度，
 * 返回前重新获取写锁。
 */
int __cond_resched_rwlock_write(rwlock_t *lock)
{
	int resched = should_resched(PREEMPT_LOCK_OFFSET);
	int ret = 0;

	lockdep_assert_held_write(lock);

	if (rwlock_needbreak(lock) || resched) {
		write_unlock(lock);
		if (!_cond_resched())
			cpu_relax();
		ret = 1;
		write_lock(lock);
	}
	return ret;
}
EXPORT_SYMBOL(__cond_resched_rwlock_write);

#ifdef CONFIG_PREEMPT_DYNAMIC

# ifdef CONFIG_GENERIC_IRQ_ENTRY
#  include <linux/irq-entry-common.h>
# endif

/*
 * SC:cond_resched
 * SC:might_resched
 * SC:preempt_schedule
 * SC:preempt_schedule_notrace
 * SC:irqentry_exit_cond_resched
 *
 *
 * NONE:
 *   cond_resched               <- __cond_resched
 *   might_resched              <- RET0
 *   preempt_schedule           <- NOP
 *   preempt_schedule_notrace   <- NOP
 *   irqentry_exit_cond_resched <- NOP
 *   dynamic_preempt_lazy       <- false
 *
 * VOLUNTARY:
 *   cond_resched               <- __cond_resched
 *   might_resched              <- __cond_resched
 *   preempt_schedule           <- NOP
 *   preempt_schedule_notrace   <- NOP
 *   irqentry_exit_cond_resched <- NOP
 *   dynamic_preempt_lazy       <- false
 *
 * FULL:
 *   cond_resched               <- RET0
 *   might_resched              <- RET0
 *   preempt_schedule           <- preempt_schedule
 *   preempt_schedule_notrace   <- preempt_schedule_notrace
 *   irqentry_exit_cond_resched <- irqentry_exit_cond_resched
 *   dynamic_preempt_lazy       <- false
 *
 * LAZY:
 *   cond_resched               <- RET0
 *   might_resched              <- RET0
 *   preempt_schedule           <- preempt_schedule
 *   preempt_schedule_notrace   <- preempt_schedule_notrace
 *   irqentry_exit_cond_resched <- irqentry_exit_cond_resched
 *   dynamic_preempt_lazy       <- true
 */

enum {
	preempt_dynamic_undefined = -1,
	preempt_dynamic_none,
	preempt_dynamic_voluntary,
	preempt_dynamic_full,
	preempt_dynamic_lazy,
};

int preempt_dynamic_mode = preempt_dynamic_undefined;

/*
 * sched_dynamic_mode - 将抢占模式名称转换为受支持的模式编号，
 * 无效名称返回 -EINVAL。
 */
int sched_dynamic_mode(const char *str)
{
# ifndef CONFIG_PREEMPT_RT
	if (!strcmp(str, "none"))
		return preempt_dynamic_none;

	if (!strcmp(str, "voluntary"))
		return preempt_dynamic_voluntary;
# endif

	if (!strcmp(str, "full"))
		return preempt_dynamic_full;

# ifdef CONFIG_ARCH_HAS_PREEMPT_LAZY
	if (!strcmp(str, "lazy"))
		return preempt_dynamic_lazy;
# endif

	return -EINVAL;
}

# define preempt_dynamic_key_enable(f)	static_key_enable(&sk_dynamic_##f.key)
# define preempt_dynamic_key_disable(f)	static_key_disable(&sk_dynamic_##f.key)

# if defined(CONFIG_HAVE_PREEMPT_DYNAMIC_CALL)
#  define preempt_dynamic_enable(f)	static_call_update(f, f##_dynamic_enabled)
#  define preempt_dynamic_disable(f)	static_call_update(f, f##_dynamic_disabled)
# elif defined(CONFIG_HAVE_PREEMPT_DYNAMIC_KEY)
#  define preempt_dynamic_enable(f)	preempt_dynamic_key_enable(f)
#  define preempt_dynamic_disable(f)	preempt_dynamic_key_disable(f)
# else
#  error "Unsupported PREEMPT_DYNAMIC mechanism"
# endif

static DEFINE_MUTEX(sched_dynamic_mutex);

/*
 * __sched_dynamic_update - 更新静态调用和静态分支以切换抢占模式，
 * 避免中间无调度状态。
 * 调用者负责序列化模式更新。
 */
static void __sched_dynamic_update(int mode)
{
	/*
	 * 修改的是在哪些内核边界响应调度请求，不是任务的调度策略。
	 * none 保留显式条件调度，voluntary 再启用 might_resched 点，
	 * full 启用抢占/IRQ 返回入口，lazy 再允许部分请求延后响应。
	 * 先全部启用安全入口，再按目标模式关闭多余入口，防止转换途中
	 * 所有响应路径都关闭；静态调用/分支让常用路径少做动态判断。
	 */
	/*
	 * Avoid {NONE,VOLUNTARY} -> FULL transitions from ever ending up in
	 * the ZERO state, which is invalid.
	 */
	preempt_dynamic_enable(cond_resched);
	preempt_dynamic_enable(might_resched);
	preempt_dynamic_enable(preempt_schedule);
	preempt_dynamic_enable(preempt_schedule_notrace);
	preempt_dynamic_enable(irqentry_exit_cond_resched);
	preempt_dynamic_key_disable(preempt_lazy);

	switch (mode) {
	case preempt_dynamic_none:
		preempt_dynamic_enable(cond_resched);
		preempt_dynamic_disable(might_resched);
		preempt_dynamic_disable(preempt_schedule);
		preempt_dynamic_disable(preempt_schedule_notrace);
		preempt_dynamic_disable(irqentry_exit_cond_resched);
		preempt_dynamic_key_disable(preempt_lazy);
		if (mode != preempt_dynamic_mode)
			pr_info("Dynamic Preempt: none\n");
		break;

	case preempt_dynamic_voluntary:
		preempt_dynamic_enable(cond_resched);
		preempt_dynamic_enable(might_resched);
		preempt_dynamic_disable(preempt_schedule);
		preempt_dynamic_disable(preempt_schedule_notrace);
		preempt_dynamic_disable(irqentry_exit_cond_resched);
		preempt_dynamic_key_disable(preempt_lazy);
		if (mode != preempt_dynamic_mode)
			pr_info("Dynamic Preempt: voluntary\n");
		break;

	case preempt_dynamic_full:
		preempt_dynamic_disable(cond_resched);
		preempt_dynamic_disable(might_resched);
		preempt_dynamic_enable(preempt_schedule);
		preempt_dynamic_enable(preempt_schedule_notrace);
		preempt_dynamic_enable(irqentry_exit_cond_resched);
		preempt_dynamic_key_disable(preempt_lazy);
		if (mode != preempt_dynamic_mode)
			pr_info("Dynamic Preempt: full\n");
		break;

	case preempt_dynamic_lazy:
		preempt_dynamic_disable(cond_resched);
		preempt_dynamic_disable(might_resched);
		preempt_dynamic_enable(preempt_schedule);
		preempt_dynamic_enable(preempt_schedule_notrace);
		preempt_dynamic_enable(irqentry_exit_cond_resched);
		preempt_dynamic_key_enable(preempt_lazy);
		if (mode != preempt_dynamic_mode)
			pr_info("Dynamic Preempt: lazy\n");
		break;
	}

	preempt_dynamic_mode = mode;
}

/*
 * sched_dynamic_update - 在互斥锁保护下切换系统的动态抢占模式。
 */
void sched_dynamic_update(int mode)
{
	mutex_lock(&sched_dynamic_mutex);
	__sched_dynamic_update(mode);
	mutex_unlock(&sched_dynamic_mutex);
}

/*
 * setup_preempt_mode - 解析 preempt 启动参数并应用受支持的抢占模式。
 */
static int __init setup_preempt_mode(char *str)
{
	int mode = sched_dynamic_mode(str);
	if (mode < 0) {
		pr_warn("Dynamic Preempt: unsupported mode: %s\n", str);
		return 0;
	}

	sched_dynamic_update(mode);
	return 1;
}
__setup("preempt=", setup_preempt_mode);

/*
 * preempt_dynamic_init - 未通过启动参数指定时，
 * 按编译配置初始化动态抢占模式。
 */
static void __init preempt_dynamic_init(void)
{
	if (preempt_dynamic_mode == preempt_dynamic_undefined) {
		if (IS_ENABLED(CONFIG_PREEMPT_NONE)) {
			sched_dynamic_update(preempt_dynamic_none);
		} else if (IS_ENABLED(CONFIG_PREEMPT_VOLUNTARY)) {
			sched_dynamic_update(preempt_dynamic_voluntary);
		} else if (IS_ENABLED(CONFIG_PREEMPT_LAZY)) {
			sched_dynamic_update(preempt_dynamic_lazy);
		} else {
			/* Default static call setting, nothing to do */
			WARN_ON_ONCE(!IS_ENABLED(CONFIG_PREEMPT));
			preempt_dynamic_mode = preempt_dynamic_full;
			pr_info("Dynamic Preempt: full\n");
		}
	}
}

/*
 * PREEMPT_MODEL_ACCESSOR - 生成 preempt_model_<mode>() 查询函数。
 * 各函数判断当前是否采用对应的 none、voluntary、full 或 lazy 抢占模式。
 */
# define PREEMPT_MODEL_ACCESSOR(mode) \
	bool preempt_model_##mode(void)						 \
	{									 \
		WARN_ON_ONCE(preempt_dynamic_mode == preempt_dynamic_undefined); \
		return preempt_dynamic_mode == preempt_dynamic_##mode;		 \
	}									 \
	EXPORT_SYMBOL_GPL(preempt_model_##mode)

/*
 * preempt_model_none - 判断当前动态抢占模式是否为非抢占模式。
 */
PREEMPT_MODEL_ACCESSOR(none);
/*
 * preempt_model_voluntary - 判断当前动态抢占模式是否为自愿抢占模式。
 */
PREEMPT_MODEL_ACCESSOR(voluntary);
/*
 * preempt_model_full - 判断当前动态抢占模式是否为完全抢占模式。
 */
PREEMPT_MODEL_ACCESSOR(full);
/*
 * preempt_model_lazy - 判断当前动态抢占模式是否为延迟抢占模式。
 */
PREEMPT_MODEL_ACCESSOR(lazy);

#else /* !CONFIG_PREEMPT_DYNAMIC: */

#define preempt_dynamic_mode -1

/*
 * preempt_dynamic_init - 未启用动态抢占时的空操作初始化接口。
 */
static inline void preempt_dynamic_init(void) { }

#endif /* CONFIG_PREEMPT_DYNAMIC */

const char *preempt_modes[] = {
	"none", "voluntary", "full", "lazy", NULL,
};

/*
 * preempt_model_str - 返回当前内核抢占模型的可读字符串，
 * 包含动态模式及 RT 等修饰信息。
 */
const char *preempt_model_str(void)
{
	bool brace = IS_ENABLED(CONFIG_PREEMPT_RT) &&
		(IS_ENABLED(CONFIG_PREEMPT_DYNAMIC) ||
		 IS_ENABLED(CONFIG_PREEMPT_LAZY));
	static char buf[128];

	if (IS_ENABLED(CONFIG_PREEMPT_BUILD)) {
		struct seq_buf s;

		seq_buf_init(&s, buf, sizeof(buf));
		seq_buf_puts(&s, "PREEMPT");

		if (IS_ENABLED(CONFIG_PREEMPT_RT))
			seq_buf_printf(&s, "%sRT%s",
				       brace ? "_{" : "_",
				       brace ? "," : "");

		if (IS_ENABLED(CONFIG_PREEMPT_DYNAMIC)) {
			seq_buf_printf(&s, "(%s)%s",
				       preempt_dynamic_mode >= 0 ?
				       preempt_modes[preempt_dynamic_mode] : "undef",
				       brace ? "}" : "");
			return seq_buf_str(&s);
		}

		if (IS_ENABLED(CONFIG_PREEMPT_LAZY)) {
			seq_buf_printf(&s, "LAZY%s",
				       brace ? "}" : "");
			return seq_buf_str(&s);
		}

		return seq_buf_str(&s);
	}

	if (IS_ENABLED(CONFIG_PREEMPT_VOLUNTARY_BUILD))
		return "VOLUNTARY";

	return "NONE";
}

/*
 * io_schedule_prepare - 设置当前任务的 I/O 等待标志并提交积压请求，
 * 返回旧标志作为令牌。
 */
int io_schedule_prepare(void)
{
	int old_iowait = current->in_iowait;

	/*
	 * in_iowait 是记账身份，不是 TASK_* 睡眠状态，调用者仍须按等待
	 * 协议设置任务状态。保存旧值用于嵌套恢复，真正阻塞/唤醒时才
	 * 由调度路径增减 rq->nr_iowait；先提交 plug 让待完成的 I/O 能推进。
	 */
	current->in_iowait = 1;
	blk_flush_plug(current->plug, true);
	return old_iowait;
}

/*
 * io_schedule_finish - 使用准备阶段保存的令牌
 * 恢复当前任务原有的 I/O 等待标志。
 */
void io_schedule_finish(int token)
{
	current->in_iowait = token;
}

/*
 * This task is about to go to sleep on IO. Increment rq->nr_iowait so
 * that process accounting knows that this is a task in IO wait state.
 */
/*
 * io_schedule_timeout - 以 I/O 等待身份进行带超时的调度，
 * 并在返回时恢复原等待标志。
 * timeout 和返回的剩余时间均以 jiffies 为单位。
 */
long __sched io_schedule_timeout(long timeout)
{
	int token;
	long ret;

	token = io_schedule_prepare();
	ret = schedule_timeout(timeout);
	io_schedule_finish(token);

	return ret;
}
EXPORT_SYMBOL(io_schedule_timeout);

/*
 * io_schedule - 在当前任务调度睡眠期间标记 I/O 等待，返回后恢复原有标志。
 */
void __sched io_schedule(void)
{
	int token;

	token = io_schedule_prepare();
	schedule();
	io_schedule_finish(token);
}
EXPORT_SYMBOL(io_schedule);

/*
 * sched_show_task - 安全获取任务栈后
 * 输出状态、进程标识、工作线程信息及调用栈。
 */
void sched_show_task(struct task_struct *p)
{
	unsigned long free;
	int ppid;

	if (!try_get_task_stack(p))
		return;

	pr_info("task:%-15.15s state:%c", p->comm, task_state_to_char(p));

	if (task_is_running(p))
		pr_cont("  running task    ");
	free = stack_not_used(p);
	ppid = 0;
	rcu_read_lock();
	if (pid_alive(p))
		ppid = task_pid_nr(rcu_dereference(p->real_parent));
	rcu_read_unlock();
	pr_cont(" stack:%-5lu pid:%-5d tgid:%-5d ppid:%-6d task_flags:0x%04x flags:0x%08lx\n",
		free, task_pid_nr(p), task_tgid_nr(p),
		ppid, p->flags, read_task_thread_flags(p));

	print_worker_info(KERN_INFO, p);
	print_stop_info(KERN_INFO, p);
	print_scx_info(KERN_INFO, p);
	show_stack(p, NULL, KERN_INFO);
	put_task_stack(p);
}
EXPORT_SYMBOL_GPL(sched_show_task);

/*
 * state_filter_match - 判断任务是否匹配状态过滤条件，
 * 查询不可中断睡眠时排除 TASK_IDLE。
 */
static inline bool
state_filter_match(unsigned long state_filter, struct task_struct *p)
{
	unsigned int state = READ_ONCE(p->__state);

	/* no filter, everything matches */
	if (!state_filter)
		return true;

	/* filter, but doesn't match */
	if (!(state & state_filter))
		return false;

	/*
	 * When looking for TASK_UNINTERRUPTIBLE skip TASK_IDLE (allows
	 * TASK_KILLABLE).
	 */
	if (state_filter == TASK_UNINTERRUPTIBLE && (state & TASK_NOLOAD))
		return false;

	return true;
}


/*
 * show_state_filter - 遍历并输出匹配状态的所有任务；
 * 无过滤时额外输出调度及锁信息。
 */
void show_state_filter(unsigned int state_filter)
{
	struct task_struct *g, *p;

	rcu_read_lock();
	for_each_process_thread(g, p) {
		/*
		 * reset the NMI-timeout, listing all files on a slow
		 * console might take a lot of time:
		 * Also, reset softlockup watchdogs on all CPUs, because
		 * another CPU might be blocked waiting for us to process
		 * an IPI.
		 */
		touch_nmi_watchdog();
		touch_all_softlockup_watchdogs();
		if (state_filter_match(state_filter, p))
			sched_show_task(p);
	}

	if (!state_filter)
		sysrq_sched_debug_show();

	rcu_read_unlock();
	/*
	 * Only show locks if all tasks are dumped:
	 */
	if (!state_filter)
		debug_show_all_locks();
}

/**
 * init_idle - set up an idle thread for a given CPU
 * 初始化指定 CPU 的空闲任务、亲和性、调度类和当前队列状态。
 * @idle: task in question
 * @cpu: CPU the idle task belongs to
 *
 * NOTE: this function does not set the idle thread's NEED_RESCHED
 * flag, to make booting more robust.
 */
void __init init_idle(struct task_struct *idle, int cpu)
{
	struct affinity_context ac = (struct affinity_context) {
		.new_mask  = cpumask_of(cpu),
		.flags     = 0,
	};
	struct rq *rq = cpu_rq(cpu);
	unsigned long flags;

	raw_spin_lock_irqsave(&idle->pi_lock, flags);
	raw_spin_rq_lock(rq);

	idle->__state = TASK_RUNNING;
	idle->se.exec_start = sched_clock();
	/*
	 * PF_KTHREAD should already be set at this point; regardless, make it
	 * look like a proper per-CPU kthread.
	 */
	idle->flags |= PF_KTHREAD | PF_NO_SETAFFINITY;
	kthread_set_per_cpu(idle, cpu);

	/*
	 * No validation and serialization required at boot time and for
	 * setting up the idle tasks of not yet online CPUs.
	 */
	set_cpus_allowed_common(idle, &ac);
	/*
	 * We're having a chicken and egg problem, even though we are
	 * holding rq->lock, the CPU isn't yet set to this CPU so the
	 * lockdep check in task_group() will fail.
	 *
	 * Similar case to sched_fork(). / Alternatively we could
	 * use task_rq_lock() here and obtain the other rq->lock.
	 *
	 * Silence PROVE_RCU
	 */
	rcu_read_lock();
	__set_task_cpu(idle, cpu);
	rcu_read_unlock();

	rq->idle = idle;
	rq_set_donor(rq, idle);
	rcu_assign_pointer(rq->curr, idle);
	/*
	 * idle 是永远可选的特殊任务：标记 QUEUED/on_cpu，但不作为普通
	 * 可运行任务增加 rq->nr_running。队列无工作时由 idle 类兜底，
	 * 因而 nr_running==0 不表示 CPU 没有执行上下文。
	 */
	idle->on_rq = TASK_ON_RQ_QUEUED;
	idle->on_cpu = 1;
	raw_spin_rq_unlock(rq);
	raw_spin_unlock_irqrestore(&idle->pi_lock, flags);

	/* Set the preempt count _outside_ the spinlocks! */
	init_idle_preempt_count(idle, cpu);

	/*
	 * The idle tasks have their own, simple scheduling class:
	 */
	idle->sched_class = &idle_sched_class;
	ftrace_graph_init_idle_task(idle, cpu);
	vtime_init_idle(idle, cpu);
	sprintf(idle->comm, "%s/%d", INIT_TASK_COMM, cpu);
}

/*
 * cpuset_cpumask_can_shrink - 检查缩减 cpuset 的 CPU 集合
 * 是否仍能满足截止时间带宽约束。
 */
int cpuset_cpumask_can_shrink(const struct cpumask *cur,
			      const struct cpumask *trial)
{
	int ret = 1;

	if (cpumask_empty(cur))
		return ret;

	ret = dl_cpuset_cpumask_can_shrink(cur, trial);

	return ret;
}

/*
 * task_can_attach - 检查任务是否允许迁入 cpuset，
 * 禁止修改亲和性的任务返回 -EINVAL。
 */
int task_can_attach(struct task_struct *p)
{
	int ret = 0;

	/*
	 * Kthreads which disallow setaffinity shouldn't be moved
	 * to a new cpuset; we don't want to change their CPU
	 * affinity and isolating such threads by their set of
	 * allowed nodes is unnecessary.  Thus, cpusets are not
	 * applicable for such threads.  This prevents checking for
	 * success of set_cpus_allowed_ptr() on all attached tasks
	 * before cpus_mask may be changed.
	 */
	if (p->flags & PF_NO_SETAFFINITY)
		ret = -EINVAL;

	return ret;
}

bool sched_smp_initialized __read_mostly;

#ifdef CONFIG_NUMA_BALANCING
/* Migrate current task p to target_cpu */
/*
 * migrate_task_to - 校验亲和性后通过源 CPU stopper
 * 将任务迁到指定 NUMA 目标 CPU。
 */
int migrate_task_to(struct task_struct *p, int target_cpu)
{
	struct migration_arg arg = { p, target_cpu };
	int curr_cpu = task_cpu(p);

	if (curr_cpu == target_cpu)
		return 0;

	if (!cpumask_test_cpu(target_cpu, p->cpus_ptr))
		return -EINVAL;

	/* TODO: This is not properly updating schedstats */

	trace_sched_move_numa(p, curr_cpu, target_cpu);
	return stop_one_cpu(curr_cpu, migration_cpu_stop, &arg);
}

/*
 * Requeue a task on a given node and accurately track the number of NUMA
 * tasks on the runqueues
 */
/*
 * sched_setnuma - 在任务及队列锁保护下修改首选 NUMA 节点，
 * 并通过重入队维护统计。
 */
void sched_setnuma(struct task_struct *p, int nid)
{
	guard(task_rq_lock)(p);
	scoped_guard (sched_change, p, DEQUEUE_SAVE)
		p->numa_preferred_nid = nid;
}
#endif /* CONFIG_NUMA_BALANCING */

#ifdef CONFIG_HOTPLUG_CPU
/*
 * Invoked on the outgoing CPU in context of the CPU hotplug thread
 * after ensuring that there are no user space tasks left on the CPU.
 *
 * If there is a lazy mm in use on the hotplug thread, drop it and
 * switch to init_mm.
 *
 * The reference count on init_mm is dropped in finish_cpu().
 */
/*
 * sched_force_init_mm - CPU 下线前让热插拔线程放弃借用的地址空间，
 * 切换到 init_mm。
 */
static void sched_force_init_mm(void)
{
	struct mm_struct *mm = current->active_mm;

	if (mm != &init_mm) {
		mmgrab_lazy_tlb(&init_mm);
		local_irq_disable();
		current->active_mm = &init_mm;
		switch_mm_irqs_off(mm, &init_mm, current);
		local_irq_enable();
		finish_arch_post_lock_switch();
		mmdrop_lazy_tlb(mm);
	}

	/* finish_cpu(), as ran on the BP, will clean up the active_mm state */
}

/*
 * __balance_push_cpu_stop - 在 stopper 上下文
 * 将待下线 CPU 上的普通任务迁往合法替代 CPU。
 */
static int __balance_push_cpu_stop(void *arg)
{
	struct task_struct *p = arg;
	struct rq *rq = this_rq();
	struct rq_flags rf;
	int cpu;

	raw_spin_lock_irq(&p->pi_lock);
	rq_lock(rq, &rf);

	update_rq_clock(rq);

	if (task_rq(p) == rq && task_on_rq_queued(p)) {
		cpu = select_fallback_rq(rq->cpu, p);
		rq = __migrate_task(rq, &rf, p, cpu);
	}

	rq_unlock(rq, &rf);
	raw_spin_unlock_irq(&p->pi_lock);

	put_task_struct(p);

	return 0;
}

static DEFINE_PER_CPU(struct cpu_stop_work, push_work);

/*
 * Ensure we only run per-cpu kthreads once the CPU goes !active.
 *
 * This is enabled below SCHED_AP_ACTIVE; when !cpu_active(), but only
 * effective when the hotplug motion is down.
 */
/*
 * balance_push - CPU 下线期间持续推走普通任务，
 * 仅允许每 CPU 线程和禁止迁移任务停留。
 * 队列排空时唤醒热插拔等待者；调用者须持有运行队列锁。
 */
static void balance_push(struct rq *rq)
{
	struct task_struct *push_task = rq->curr;

	/*
	 * 下线不是一次性搬空：正在运行和禁止迁移的任务可能暂时留下。
	 * 把回调持续挂在调度收尾处，每次切换再尝试推走当前普通任务；
	 * 最终由 wait_empty 等待排空，再让 dying 回调核验并结算统计。
	 */
	lockdep_assert_rq_held(rq);

	/*
	 * Ensure the thing is persistent until balance_push_set(.on = false);
	 */
	rq->balance_callback = &balance_push_callback;

	/*
	 * Only active while going offline and when invoked on the outgoing
	 * CPU.
	 */
	if (!cpu_dying(rq->cpu) || rq != this_rq())
		return;

	/*
	 * Both the cpu-hotplug and stop task are in this case and are
	 * required to complete the hotplug process.
	 */
	if (kthread_is_per_cpu(push_task) ||
	    is_migration_disabled(push_task)) {

		/*
		 * If this is the idle task on the outgoing CPU try to wake
		 * up the hotplug control thread which might wait for the
		 * last task to vanish. The rcuwait_active() check is
		 * accurate here because the waiter is pinned on this CPU
		 * and can't obviously be running in parallel.
		 *
		 * On RT kernels this also has to check whether there are
		 * pinned and scheduled out tasks on the runqueue. They
		 * need to leave the migrate disabled section first.
		 */
		if (!rq->nr_running && !rq_has_pinned_tasks(rq) &&
		    rcuwait_active(&rq->hotplug_wait)) {
			raw_spin_rq_unlock(rq);
			rcuwait_wake_up(&rq->hotplug_wait);
			raw_spin_rq_lock(rq);
		}
		return;
	}

	get_task_struct(push_task);
	/*
	 * Temporarily drop rq->lock such that we can wake-up the stop task.
	 * Both preemption and IRQs are still disabled.
	 */
	preempt_disable();
	raw_spin_rq_unlock(rq);
	stop_one_cpu_nowait(rq->cpu, __balance_push_cpu_stop, push_task,
			    this_cpu_ptr(&push_work));
	preempt_enable();
	/*
	 * At this point need_resched() is true and we'll take the loop in
	 * schedule(). The next pick is obviously going to be the stop task
	 * which kthread_is_per_cpu() and will push this task away.
	 */
	raw_spin_rq_lock(rq);
}

/*
 * balance_push_set - 加锁安装或移除指定 CPU 的持久热插拔任务推送回调。
 */
static void balance_push_set(int cpu, bool on)
{
	struct rq *rq = cpu_rq(cpu);
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);
	if (on) {
		WARN_ON_ONCE(rq->balance_callback);
		rq->balance_callback = &balance_push_callback;
	} else if (rq->balance_callback == &balance_push_callback) {
		rq->balance_callback = NULL;
	}
	rq_unlock_irqrestore(rq, &rf);
}

/*
 * Invoked from a CPUs hotplug control thread after the CPU has been marked
 * inactive. All tasks which are not per CPU kernel threads are either
 * pushed off this CPU now via balance_push() or placed on a different CPU
 * during wakeup. Wait until the CPU is quiescent.
 */
/*
 * balance_hotplug_wait - 等待本 CPU 只剩热插拔控制线程，
 * 且不再有禁止迁移的任务。
 */
static void balance_hotplug_wait(void)
{
	struct rq *rq = this_rq();

	rcuwait_wait_event(&rq->hotplug_wait,
			   rq->nr_running == 1 && !rq_has_pinned_tasks(rq),
			   TASK_UNINTERRUPTIBLE);
}

#else /* !CONFIG_HOTPLUG_CPU: */

/*
 * balance_push - 未启用 CPU 热插拔时的空操作任务推送接口。
 */
static inline void balance_push(struct rq *rq)
{
}

/*
 * balance_push_set - 未启用 CPU 热插拔时的空操作推送开关接口。
 */
static inline void balance_push_set(int cpu, bool on)
{
}

/*
 * balance_hotplug_wait - 未启用 CPU 热插拔时的空操作排空等待接口。
 */
static inline void balance_hotplug_wait(void)
{
}

#endif /* !CONFIG_HOTPLUG_CPU */

/*
 * set_rq_online - 将运行队列加入根域在线集合，并通知各调度类队列已上线。
 */
void set_rq_online(struct rq *rq)
{
	if (!rq->online) {
		const struct sched_class *class;

		cpumask_set_cpu(rq->cpu, rq->rd->online);
		rq->online = 1;

		for_each_class(class) {
			if (class->rq_online)
				class->rq_online(rq);
		}
	}
}

/*
 * set_rq_offline - 更新队列时钟、通知各调度类下线，
 * 并从根域在线集合移除队列。
 */
void set_rq_offline(struct rq *rq)
{
	if (rq->online) {
		const struct sched_class *class;

		update_rq_clock(rq);
		for_each_class(class) {
			if (class->rq_offline)
				class->rq_offline(rq);
		}

		cpumask_clear_cpu(rq->cpu, rq->rd->online);
		rq->online = 0;
	}
}

/*
 * sched_set_rq_online - 保存中断状态并加锁，确认根域有效后将运行队列上线。
 */
static inline void sched_set_rq_online(struct rq *rq, int cpu)
{
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);
	if (rq->rd) {
		BUG_ON(!cpumask_test_cpu(cpu, rq->rd->span));
		set_rq_online(rq);
	}
	rq_unlock_irqrestore(rq, &rf);
}

/*
 * sched_set_rq_offline - 保存中断状态并加锁，
 * 确认根域有效后将运行队列下线。
 */
static inline void sched_set_rq_offline(struct rq *rq, int cpu)
{
	struct rq_flags rf;

	rq_lock_irqsave(rq, &rf);
	if (rq->rd) {
		BUG_ON(!cpumask_test_cpu(cpu, rq->rd->span));
		set_rq_offline(rq);
	}
	rq_unlock_irqrestore(rq, &rf);
}

/*
 * used to mark begin/end of suspend/resume:
 */
static int num_cpus_frozen;

/*
 * Update cpusets according to cpu_active mask.  If cpusets are
 * disabled, cpuset_update_active_cpus() becomes a simple wrapper
 * around partition_sched_domains().
 *
 * If we come here as part of a suspend/resume, don't touch cpusets because we
 * want to restore it back to its original state upon resume anyway.
 */
/*
 * cpuset_cpu_active - CPU 激活后更新 cpuset 与调度域，
 * 系统恢复时延后重建完整分区。
 */
static void cpuset_cpu_active(void)
{
	if (cpuhp_tasks_frozen) {
		/*
		 * num_cpus_frozen tracks how many CPUs are involved in suspend
		 * resume sequence. As long as this is not the last online
		 * operation in the resume sequence, just build a single sched
		 * domain, ignoring cpusets.
		 */
		cpuset_reset_sched_domains();
		if (--num_cpus_frozen)
			return;
		/*
		 * This is the last CPU online operation. So fall through and
		 * restore the original sched domains by considering the
		 * cpuset configurations.
		 */
		cpuset_force_rebuild();
	}
	cpuset_update_active_cpus();
}

/*
 * cpuset_cpu_inactive - CPU 停用后更新 cpuset；
 * 挂起期间记录冻结数量并临时重置调度域。
 */
static void cpuset_cpu_inactive(unsigned int cpu)
{
	if (!cpuhp_tasks_frozen) {
		cpuset_update_active_cpus();
	} else {
		num_cpus_frozen++;
		cpuset_reset_sched_domains();
	}
}

/*
 * sched_smt_present_inc - CPU 上线使核心出现 SMT 并行线程时，
 * 增加 SMT 存在静态分支引用。
 */
static inline void sched_smt_present_inc(int cpu)
{
#ifdef CONFIG_SCHED_SMT
	if (cpumask_weight(cpu_smt_mask(cpu)) == 2)
		static_branch_inc_cpuslocked(&sched_smt_present);
#endif
}

/*
 * sched_smt_present_dec - CPU 下线使核心失去 SMT 并行线程时，
 * 减少 SMT 存在静态分支引用。
 */
static inline void sched_smt_present_dec(int cpu)
{
#ifdef CONFIG_SCHED_SMT
	if (cpumask_weight(cpu_smt_mask(cpu)) == 2)
		static_branch_dec_cpuslocked(&sched_smt_present);
#endif
}

/*
 * sched_cpu_activate - 将 CPU 纳入正常调度，
 * 更新 SMT、NUMA、cpuset、扩展调度及队列状态。
 */
int sched_cpu_activate(unsigned int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	/*
	 * Clear the balance_push callback and prepare to schedule
	 * regular tasks.
	 */
	balance_push_set(cpu, false);

	/*
	 * When going up, increment the number of cores with SMT present.
	 */
	sched_smt_present_inc(cpu);
	set_cpu_active(cpu, true);

	if (sched_smp_initialized) {
		sched_update_numa(cpu, true);
		sched_domains_numa_masks_set(cpu);
		cpuset_cpu_active();
	}

	scx_rq_activate(rq);

	/*
	 * Put the rq online, if not already. This happens:
	 *
	 * 1) In the early boot process, because we build the real domains
	 *    after all CPUs have been brought up.
	 *
	 * 2) At runtime, if cpuset_cpu_active() fails to rebuild the
	 *    domains.
	 */
	sched_set_rq_online(rq, cpu);

	return 0;
}

/*
 * sched_cpu_deactivate - 校验截止时间带宽后停用 CPU，
 * 阻止新任务进入并启动任务迁出。
 * 经 RCU 同步后更新队列、SMT、NUMA 和 cpuset 状态，失败返回错误码。
 */
int sched_cpu_deactivate(unsigned int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	int ret;

	ret = dl_bw_deactivate(cpu);

	if (ret)
		return ret;

	/*
	 * Remove CPU from nohz.idle_cpus_mask to prevent participating in
	 * load balancing when not active
	 */
	nohz_balance_exit_idle(rq);

	/*
	 * 先阻止新的普通任务放置，再排出旧任务。清 active 不等于立即
	 * 停止执行，CPU 仍在线，才能运行 stopper 和迁移禁用区的收尾。
	 */
	set_cpu_active(cpu, false);

	/*
	 * From this point forward, this CPU will refuse to run any task that
	 * is not: migrate_disable() or KTHREAD_IS_PER_CPU, and will actively
	 * push those tasks away until this gets cleared, see
	 * sched_cpu_dying().
	 */
	balance_push_set(cpu, true);

	/*
	 * We've cleared cpu_active_mask / set balance_push, wait for all
	 * preempt-disabled and RCU users of this state to go away such that
	 * all new such users will observe it.
	 *
	 * Specifically, we rely on ttwu to no longer target this CPU, see
	 * ttwu_queue_cond() and is_cpu_allowed().
	 *
	 * Do sync before park smpboot threads to take care the RCU boost case.
	 */
	synchronize_rcu();

	/* 旧选核读者退出后，再向各调度类宣告本 rq 不再承接工作。 */
	sched_set_rq_offline(rq, cpu);

	scx_rq_deactivate(rq);

	/*
	 * When going down, decrement the number of cores with SMT present.
	 */
	sched_smt_present_dec(cpu);

#ifdef CONFIG_SCHED_SMT
	sched_core_cpu_deactivate(cpu);
#endif

	if (!sched_smp_initialized)
		return 0;

	sched_update_numa(cpu, false);
	cpuset_cpu_inactive(cpu);
	sched_domains_numa_masks_clear(cpu);
	return 0;
}

/*
 * sched_rq_cpu_starting - CPU 启动时同步负载采样时间，并更新最大均衡间隔。
 */
static void sched_rq_cpu_starting(unsigned int cpu)
{
	struct rq *rq = cpu_rq(cpu);

	rq->calc_load_update = calc_load_update;
	update_max_interval();
}

/*
 * sched_cpu_starting - CPU 启动回调，
 * 初始化核心共享队列状态及负载时间并启动远程 tick。
 */
int sched_cpu_starting(unsigned int cpu)
{
	sched_core_cpu_starting(cpu);
	sched_rq_cpu_starting(cpu);
	sched_tick_start(cpu);
	return 0;
}

#ifdef CONFIG_HOTPLUG_CPU

/*
 * Invoked immediately before the stopper thread is invoked to bring the
 * CPU down completely. At this point all per CPU kthreads except the
 * hotplug thread (current) and the stopper thread (inactive) have been
 * either parked or have been unbound from the outgoing CPU. Ensure that
 * any of those which might be on the way out are gone.
 *
 * If after this point a bound task is being woken on this CPU then the
 * responsible hotplug callback has failed to do it's job.
 * sched_cpu_dying() will catch it with the appropriate fireworks.
 */
/*
 * sched_cpu_wait_empty - CPU 最终停止前等待普通任务迁出，
 * 并将热插拔线程切到 init_mm。
 */
int sched_cpu_wait_empty(unsigned int cpu)
{
	balance_hotplug_wait();
	sched_force_init_mm();
	return 0;
}

/*
 * Since this CPU is going 'away' for a while, fold any nr_active delta we
 * might have. Called from the CPU stopper task after ensuring that the
 * stopper is the last running task on the CPU, so nr_active count is
 * stable. We need to take the tear-down thread which is calling this into
 * account, so we hand in adjust = 1 to the load calculation.
 *
 * Also see the comment "Global load-average calculations".
 */
/*
 * calc_load_migrate - CPU 下线前将其未折算的活动任务数增量
 * 合并到全局负载统计。
 */
static void calc_load_migrate(struct rq *rq)
{
	long delta = calc_load_fold_active(rq, 1);

	if (delta)
		atomic_long_add(delta, &calc_load_tasks);
}

/*
 * dump_rq_tasks - 持有队列锁时按指定日志级别输出该 CPU 上仍已排队的任务。
 */
static void dump_rq_tasks(struct rq *rq, const char *loglvl)
{
	struct task_struct *g, *p;
	int cpu = cpu_of(rq);

	lockdep_assert_rq_held(rq);

	printk("%sCPU%d enqueued tasks (%u total):\n", loglvl, cpu, rq->nr_running);
	for_each_process_thread(g, p) {
		if (task_cpu(p) != cpu)
			continue;

		if (!task_on_rq_queued(p))
			continue;

		printk("%s\tpid: %d, name: %s\n", loglvl, p->pid, p->comm);
	}
}

/*
 * sched_cpu_dying - CPU 最终下线时停止 tick 和调度定时器，
 * 检查队列排空并结算负载。
 */
int sched_cpu_dying(unsigned int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	struct rq_flags rf;

	/* Handle pending wakeups and then migrate everything off */
	sched_tick_stop(cpu);

	rq_lock_irqsave(rq, &rf);
	update_rq_clock(rq);
	if (rq->nr_running != 1 || rq_has_pinned_tasks(rq)) {
		WARN(true, "Dying CPU not properly vacated!");
		dump_rq_tasks(rq, KERN_WARNING);
	}
	dl_server_stop(&rq->fair_server);
	rq_unlock_irqrestore(rq, &rf);

	calc_load_migrate(rq);
	update_max_interval();
	hrtick_clear(rq);
	sched_core_cpu_dying(cpu);
	return 0;
}
#endif /* CONFIG_HOTPLUG_CPU */

/*
 * sched_init_smp - 启动阶段建立 SMP 调度域和 NUMA 拓扑，
 * 初始化实时及截止时间调度设施。
 */
void __init sched_init_smp(void)
{
	sched_init_numa(NUMA_NO_NODE);

	prandom_init_once(&sched_rnd_state);

	/*
	 * There's no userspace yet to cause hotplug operations; hence all the
	 * CPU masks are stable and all blatant races in the below code cannot
	 * happen.
	 */
	sched_domains_mutex_lock();
	sched_init_domains(cpu_active_mask);
	sched_domains_mutex_unlock();

	/* Move init over to a non-isolated CPU */
	if (set_cpus_allowed_ptr(current, housekeeping_cpumask(HK_TYPE_DOMAIN)) < 0)
		BUG();
	current->flags &= ~PF_NO_SETAFFINITY;
	sched_init_granularity();

	init_sched_rt_class();
	init_sched_dl_class();

	sched_init_dl_servers();

	sched_smp_initialized = true;
}

/*
 * migration_init - 在早期初始化阶段执行引导 CPU 的调度启动回调。
 */
static int __init migration_init(void)
{
	sched_cpu_starting(smp_processor_id());
	return 0;
}
early_initcall(migration_init);

/*
 * in_sched_functions - 判断指令地址是否位于调度或锁实现代码区，
 * 供栈回溯等路径过滤。
 */
int in_sched_functions(unsigned long addr)
{
	return in_lock_functions(addr) ||
		(addr >= (unsigned long)__sched_text_start
		&& addr < (unsigned long)__sched_text_end);
}

#ifdef CONFIG_CGROUP_SCHED
/*
 * Default task group.
 * Every task in system belongs to this group at bootup.
 */
struct task_group root_task_group;
LIST_HEAD(task_groups);

/* Cacheline aligned slab cache for task_group */
static struct kmem_cache *task_group_cache __ro_after_init;
#endif

/*
 * sched_init - 初始化调度器全局结构、各 CPU 运行队列、根任务组，
 * 以及引导空闲任务。
 * 同时建立调度类、PSI、利用率钳制和抢占模式，
 * 最后发布调度器可运行状态。
 */
void __init sched_init(void)
{
	unsigned long ptr = 0;
	int i;

	/*
	 * 类的先后由链接器布局确定，sched_class_above 比较该顺序；
	 * 必须先确认 stop > DL > RT > fair > idle（有 ext 时插在 fair 后），
	 * 后面的“找到首个可用任务即返回”才能成立。
	 */
	/* Make sure the linker didn't screw up */
	BUG_ON(!sched_class_above(&stop_sched_class, &dl_sched_class));
	BUG_ON(!sched_class_above(&dl_sched_class, &rt_sched_class));
	BUG_ON(!sched_class_above(&rt_sched_class, &fair_sched_class));
	BUG_ON(!sched_class_above(&fair_sched_class, &idle_sched_class));
#ifdef CONFIG_SCHED_CLASS_EXT
	BUG_ON(!sched_class_above(&fair_sched_class, &ext_sched_class));
	BUG_ON(!sched_class_above(&ext_sched_class, &idle_sched_class));
#endif

	wait_bit_init();

#ifdef CONFIG_FAIR_GROUP_SCHED
	ptr += 2 * nr_cpu_ids * sizeof(void **);
#endif
#ifdef CONFIG_RT_GROUP_SCHED
	ptr += 2 * nr_cpu_ids * sizeof(void **);
#endif
	if (ptr) {
		ptr = (unsigned long)kzalloc(ptr, GFP_NOWAIT);

#ifdef CONFIG_FAIR_GROUP_SCHED
		root_task_group.se = (struct sched_entity **)ptr;
		ptr += nr_cpu_ids * sizeof(void **);

		root_task_group.cfs_rq = (struct cfs_rq **)ptr;
		ptr += nr_cpu_ids * sizeof(void **);

		root_task_group.shares = ROOT_TASK_GROUP_LOAD;
		init_cfs_bandwidth(&root_task_group.cfs_bandwidth, NULL);
#endif /* CONFIG_FAIR_GROUP_SCHED */
#ifdef CONFIG_EXT_GROUP_SCHED
		scx_tg_init(&root_task_group);
#endif /* CONFIG_EXT_GROUP_SCHED */
#ifdef CONFIG_RT_GROUP_SCHED
		root_task_group.rt_se = (struct sched_rt_entity **)ptr;
		ptr += nr_cpu_ids * sizeof(void **);

		root_task_group.rt_rq = (struct rt_rq **)ptr;
		ptr += nr_cpu_ids * sizeof(void **);

#endif /* CONFIG_RT_GROUP_SCHED */
	}

	init_defrootdomain();

#ifdef CONFIG_RT_GROUP_SCHED
	init_rt_bandwidth(&root_task_group.rt_bandwidth,
			global_rt_period(), global_rt_runtime());
#endif /* CONFIG_RT_GROUP_SCHED */

#ifdef CONFIG_CGROUP_SCHED
	task_group_cache = KMEM_CACHE(task_group, 0);

	list_add(&root_task_group.list, &task_groups);
	INIT_LIST_HEAD(&root_task_group.children);
	INIT_LIST_HEAD(&root_task_group.siblings);
	autogroup_init(&init_task);
#endif /* CONFIG_CGROUP_SCHED */

	/*
	 * 为 possible CPU 初始化，即使它尚未 online；后续热插拔上线
	 * 直接使用这些 rq、类子队列、定时器和统计，无需在切换时分配。
	 */
	for_each_possible_cpu(i) {
		struct rq *rq;

		rq = cpu_rq(i);
		raw_spin_lock_init(&rq->__lock);
		rq->nr_running = 0;
		rq->calc_load_active = 0;
		rq->calc_load_update = jiffies + LOAD_FREQ;
		init_cfs_rq(&rq->cfs);
		init_rt_rq(&rq->rt);
		init_dl_rq(&rq->dl);
#ifdef CONFIG_FAIR_GROUP_SCHED
		INIT_LIST_HEAD(&rq->leaf_cfs_rq_list);
		rq->tmp_alone_branch = &rq->leaf_cfs_rq_list;
		/*
		 * How much CPU bandwidth does root_task_group get?
		 *
		 * In case of task-groups formed through the cgroup filesystem, it
		 * gets 100% of the CPU resources in the system. This overall
		 * system CPU resource is divided among the tasks of
		 * root_task_group and its child task-groups in a fair manner,
		 * based on each entity's (task or task-group's) weight
		 * (se->load.weight).
		 *
		 * In other words, if root_task_group has 10 tasks of weight
		 * 1024) and two child groups A0 and A1 (of weight 1024 each),
		 * then A0's share of the CPU resource is:
		 *
		 *	A0's bandwidth = 1024 / (10*1024 + 1024 + 1024) = 8.33%
		 *
		 * We achieve this by letting root_task_group's tasks sit
		 * directly in rq->cfs (i.e root_task_group->se[] = NULL).
		 */
		init_tg_cfs_entry(&root_task_group, &rq->cfs, NULL, i, NULL);
#endif /* CONFIG_FAIR_GROUP_SCHED */

#ifdef CONFIG_RT_GROUP_SCHED
		/*
		 * This is required for init cpu because rt.c:__enable_runtime()
		 * starts working after scheduler_running, which is not the case
		 * yet.
		 */
		rq->rt.rt_runtime = global_rt_runtime();
		init_tg_rt_entry(&root_task_group, &rq->rt, NULL, i, NULL);
#endif
		rq->sd = NULL;
		rq->rd = NULL;
		rq->cpu_capacity = SCHED_CAPACITY_SCALE;
		rq->balance_callback = &balance_push_callback;
		rq->active_balance = 0;
		rq->next_balance = jiffies;
		rq->push_cpu = 0;
		rq->cpu = i;
		rq->online = 0;
		rq->idle_stamp = 0;
		rq->avg_idle = 2*sysctl_sched_migration_cost;
		rq->max_idle_balance_cost = sysctl_sched_migration_cost;

		INIT_LIST_HEAD(&rq->cfs_tasks);

		rq_attach_root(rq, &def_root_domain);
#ifdef CONFIG_NO_HZ_COMMON
		rq->last_blocked_load_update_tick = jiffies;
		atomic_set(&rq->nohz_flags, 0);

		INIT_CSD(&rq->nohz_csd, nohz_csd_func, rq);
#endif
#ifdef CONFIG_HOTPLUG_CPU
		rcuwait_init(&rq->hotplug_wait);
#endif
		hrtick_rq_init(rq);
		atomic_set(&rq->nr_iowait, 0);
		fair_server_init(rq);

#ifdef CONFIG_SCHED_CORE
		rq->core = rq;
		rq->core_pick = NULL;
		rq->core_dl_server = NULL;
		rq->core_enabled = 0;
		rq->core_tree = RB_ROOT;
		rq->core_forceidle_count = 0;
		rq->core_forceidle_occupation = 0;
		rq->core_forceidle_start = 0;
		rq->core_pick_in_flight = 0;

		rq->core_cookie = 0UL;
#endif
		zalloc_cpumask_var_node(&rq->scratch_mask, GFP_KERNEL, cpu_to_node(i));
	}

	set_load_weight(&init_task, false);
	init_task.se.slice = sysctl_sched_base_slice,

	/*
	 * The boot idle thread does lazy MMU switching as well:
	 */
	mmgrab_lazy_tlb(&init_mm);
	enter_lazy_tlb(&init_mm, current);

	/*
	 * The idle task doesn't need the kthread struct to function, but it
	 * is dressed up as a per-CPU kthread and thus needs to play the part
	 * if we want to avoid special-casing it in code that deals with per-CPU
	 * kthreads.
	 */
	WARN_ON(!set_kthread_struct(current));

	/*
	 * Make us the idle thread. Technically, schedule() should not be
	 * called from this thread, however somewhere below it might be,
	 * but because we are the idle thread, we just pick up running again
	 * when this runqueue becomes "idle".
	 */
	__sched_fork(0, current);
	init_idle(current, smp_processor_id());

	calc_load_update = jiffies + LOAD_FREQ;

	idle_thread_set_boot_cpu();

	balance_push_set(smp_processor_id(), false);
	init_sched_fair_class();
	init_sched_ext_class();

	psi_init();

	init_uclamp();

	preempt_dynamic_init();

	/*
	 * 基础 rq/idle/类设施就绪后才发布运行标记。SMP 拓扑和调度域
	 * 在稍后的 sched_init_smp 建立，不能把该标记当 SMP 初始化完成。
	 */
	scheduler_running = 1;
}

#ifdef CONFIG_DEBUG_ATOMIC_SLEEP

/*
 * __might_sleep - 检查可能睡眠的调用是否破坏任务状态，
 * 并验证当前上下文允许调度。
 */
void __might_sleep(const char *file, int line)
{
	unsigned int state = get_current_state();
	/*
	 * Blocking primitives will set (and therefore destroy) current->state,
	 * since we will exit with TASK_RUNNING make sure we enter with it,
	 * otherwise we will destroy state.
	 */
	WARN_ONCE(state != TASK_RUNNING && current->task_state_change,
			"do not call blocking ops when !TASK_RUNNING; "
			"state=%x set at [<%p>] %pS\n", state,
			(void *)current->task_state_change,
			(void *)current->task_state_change);

	__might_resched(file, line, 0);
}
EXPORT_SYMBOL(__might_sleep);

/*
 * print_preempt_disable_ip - 抢占计数偏离预期时输出禁止抢占的调用位置。
 */
static void print_preempt_disable_ip(int preempt_offset, unsigned long ip)
{
	if (!IS_ENABLED(CONFIG_DEBUG_PREEMPT))
		return;

	if (preempt_count() == preempt_offset)
		return;

	pr_err("Preemption disabled at:");
	print_ip_sym(KERN_ERR, ip);
}

/*
 * resched_offsets_ok - 将抢占和 RCU 嵌套深度合并，
 * 判断是否符合允许调度的预期偏移。
 */
static inline bool resched_offsets_ok(unsigned int offsets)
{
	unsigned int nested = preempt_count();

	nested += rcu_preempt_depth() << MIGHT_RESCHED_RCU_SHIFT;

	return nested == offsets;
}

/*
 * __might_resched - 检查当前上下文是否允许睡眠，
 * 违规时限频输出位置、锁及调用栈。
 */
void __might_resched(const char *file, int line, unsigned int offsets)
{
	/*
	 * 这是调用上下文诊断，不负责发起调度。offsets 描述调用点允许
	 * 存在的抢占/RCU 嵌套量；比较实际计数而非只判断是否为零，
	 * 才能处理某些封装内部自带计数偏移的合法调用点。
	 */
	/* Ratelimiting timestamp: */
	static unsigned long prev_jiffy;

	unsigned long preempt_disable_ip;

	/* WARN_ON_ONCE() by default, no rate limit required: */
	rcu_sleep_check();

	if ((resched_offsets_ok(offsets) && !irqs_disabled() &&
	     !is_idle_task(current) && !current->non_block_count) ||
	    system_state == SYSTEM_BOOTING || system_state > SYSTEM_RUNNING ||
	    oops_in_progress)
		return;

	if (time_before(jiffies, prev_jiffy + HZ) && prev_jiffy)
		return;
	prev_jiffy = jiffies;

	/* Save this before calling printk(), since that will clobber it: */
	preempt_disable_ip = get_preempt_disable_ip(current);

	pr_err("BUG: sleeping function called from invalid context at %s:%d\n",
	       file, line);
	pr_err("in_atomic(): %d, irqs_disabled(): %d, non_block: %d, pid: %d, name: %s\n",
	       in_atomic(), irqs_disabled(), current->non_block_count,
	       current->pid, current->comm);
	pr_err("preempt_count: %x, expected: %x\n", preempt_count(),
	       offsets & MIGHT_RESCHED_PREEMPT_MASK);

	if (IS_ENABLED(CONFIG_PREEMPT_RCU)) {
		pr_err("RCU nest depth: %d, expected: %u\n",
		       rcu_preempt_depth(), offsets >> MIGHT_RESCHED_RCU_SHIFT);
	}

	if (task_stack_end_corrupted(current))
		pr_emerg("Thread overran stack, or stack corrupted\n");

	debug_show_held_locks(current);
	if (irqs_disabled())
		print_irqtrace_events(current);

	print_preempt_disable_ip(offsets & MIGHT_RESCHED_PREEMPT_MASK,
				 preempt_disable_ip);

	dump_stack();
	add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
}
EXPORT_SYMBOL(__might_resched);

/*
 * __cant_sleep - 验证调用点确实处于不可睡眠的原子上下文，
 * 违例时输出诊断信息。
 */
void __cant_sleep(const char *file, int line, int preempt_offset)
{
	static unsigned long prev_jiffy;

	if (irqs_disabled())
		return;

	if (!IS_ENABLED(CONFIG_PREEMPT_COUNT))
		return;

	if (preempt_count() > preempt_offset)
		return;

	if (time_before(jiffies, prev_jiffy + HZ) && prev_jiffy)
		return;
	prev_jiffy = jiffies;

	printk(KERN_ERR "BUG: assuming atomic context at %s:%d\n", file, line);
	printk(KERN_ERR "in_atomic(): %d, irqs_disabled(): %d, pid: %d, name: %s\n",
			in_atomic(), irqs_disabled(),
			current->pid, current->comm);

	debug_show_held_locks(current);
	dump_stack();
	add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
}
EXPORT_SYMBOL_GPL(__cant_sleep);

# ifdef CONFIG_SMP
/*
 * __cant_migrate - 验证当前任务不会迁移，
 * 缺少关中断、禁止抢占或禁止迁移保护时告警。
 */
void __cant_migrate(const char *file, int line)
{
	static unsigned long prev_jiffy;

	if (irqs_disabled())
		return;

	if (is_migration_disabled(current))
		return;

	if (!IS_ENABLED(CONFIG_PREEMPT_COUNT))
		return;

	if (preempt_count() > 0)
		return;

	if (time_before(jiffies, prev_jiffy + HZ) && prev_jiffy)
		return;
	prev_jiffy = jiffies;

	pr_err("BUG: assuming non migratable context at %s:%d\n", file, line);
	pr_err("in_atomic(): %d, irqs_disabled(): %d, migration_disabled() %u pid: %d, name: %s\n",
	       in_atomic(), irqs_disabled(), is_migration_disabled(current),
	       current->pid, current->comm);

	debug_show_held_locks(current);
	dump_stack();
	add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
}
EXPORT_SYMBOL_GPL(__cant_migrate);
# endif /* CONFIG_SMP */
#endif /* CONFIG_DEBUG_ATOMIC_SLEEP */

#ifdef CONFIG_MAGIC_SYSRQ
/*
 * normalize_rt_tasks - 系统请求恢复时
 * 将用户实时和截止时间任务降为普通策略，并重置负 nice。
 */
void normalize_rt_tasks(void)
{
	struct task_struct *g, *p;
	struct sched_attr attr = {
		.sched_policy = SCHED_NORMAL,
	};

	read_lock(&tasklist_lock);
	for_each_process_thread(g, p) {
		/*
		 * Only normalize user tasks:
		 */
		if (p->flags & PF_KTHREAD)
			continue;

		p->se.exec_start = 0;
		schedstat_set(p->stats.wait_start,  0);
		schedstat_set(p->stats.sleep_start, 0);
		schedstat_set(p->stats.block_start, 0);

		if (!rt_or_dl_task(p)) {
			/*
			 * Renice negative nice level userspace
			 * tasks back to 0:
			 */
			if (task_nice(p) < 0)
				set_user_nice(p, 0);
			continue;
		}

		__sched_setscheduler(p, &attr, false, false);
	}
	read_unlock(&tasklist_lock);
}

#endif /* CONFIG_MAGIC_SYSRQ */

#ifdef CONFIG_KGDB_KDB
/*
 * These functions are only useful for KDB.
 *
 * They can only be called when the whole system has been
 * stopped - every CPU needs to be quiescent, and no scheduling
 * activity can take place. Using them for anything else would
 * be a serious bug, and as a result, they aren't even visible
 * under any other configuration.
 */

/**
 * curr_task - return the current task for a given CPU.
 * 为 KDB 返回指定 CPU 的当前任务，仅可在整个系统已停止调度时使用。
 * @cpu: the processor in question.
 *
 * ONLY VALID WHEN THE WHOLE SYSTEM IS STOPPED!
 *
 * Return: The current task for @cpu.
 */
struct task_struct *curr_task(int cpu)
{
	return cpu_curr(cpu);
}

#endif /* CONFIG_KGDB_KDB */

#ifdef CONFIG_CGROUP_SCHED
/* task_group_lock serializes the addition/removal of task groups */
static DEFINE_SPINLOCK(task_group_lock);

/*
 * task_group 是 CPU cgroup 的调度表示，不是一条全局任务队列。
 * 每组可在每个 CPU 上拥有 cfs_rq/rt_rq 及代表该组的调度实体；
 * 任务竞争组内资源，组实体再在父组中竞争，形成层级调度。
 * task_group_lock 保护层级发布/摘除，RCU 保护遍历期间的对象存活，
 * 各 rq 锁保护相应 CPU 的实体和队列；组锁不能代替 rq 锁。
 *
 * CPU cgroup 的正常生命周期可沿回调分成两条线阅读：
 *   创建：css_alloc -> sched_create_group -> css_online -> sched_online_group。
 *   回收：css_released 摘除层级 -> cgroup 提供 RCU 宽限期
 *         -> css_free 直接注销类资源 -> RCU 回调释放内存。
 * 其他使用 sched_destroy_group() 的路径由它安排第一次 RCU 回调，
 * 再进入共同的注销/延迟释放流程，不能给 css_free 重复套这一步。
 * 分配成功并不等于已向遍历者发布；从层级摘除也不等于旧读者已退出。
 */

/*
 * alloc_uclamp_sched_group - 初始化新任务组的钳制请求，
 * 并继承父组的有效钳制值。
 */
static inline void alloc_uclamp_sched_group(struct task_group *tg,
					    struct task_group *parent)
{
#ifdef CONFIG_UCLAMP_TASK_GROUP
	enum uclamp_id clamp_id;

	for_each_clamp_id(clamp_id) {
		uclamp_se_set(&tg->uclamp_req[clamp_id],
			      uclamp_none(clamp_id), false);
		tg->uclamp[clamp_id] = parent->uclamp[clamp_id];
	}
#endif
}

/*
 * sched_free_group - 释放任务组的公平、实时、自动分组资源及组对象本身。
 */
static void sched_free_group(struct task_group *tg)
{
	free_fair_sched_group(tg);
	free_rt_sched_group(tg);
	autogroup_free(tg);
	kmem_cache_free(task_group_cache, tg);
}

/*
 * sched_free_group_rcu - RCU 宽限期结束后执行任务组资源的最终释放。
 */
static void sched_free_group_rcu(struct rcu_head *rcu)
{
	sched_free_group(container_of(rcu, struct task_group, rcu));
}

/*
 * sched_unregister_group - 注销任务组的调度类资源，
 * 再等待一次 RCU 宽限期后释放对象。
 */
static void sched_unregister_group(struct task_group *tg)
{
	/*
	 * 摘掉组层级后先等待旧层级读者，再注销类队列；注销后还要等
	 * 可能通过调试/统计接口访问旧 cfs_rq 的读者，才可真正释放。
	 * 因而“从树中消失”和“内存可回收”是不同阶段。
	 */
	unregister_fair_sched_group(tg);
	unregister_rt_sched_group(tg);
	/*
	 * We have to wait for yet another RCU grace period to expire, as
	 * print_cfs_stats() might run concurrently.
	 */
	call_rcu(&tg->rcu, sched_free_group_rcu);
}

/* allocate runqueue etc for a new task group */
/*
 * sched_create_group - 分配并初始化新任务组的各调度类及利用率钳制资源。
 * 成功返回任务组指针，失败清理已分配资源并返回错误指针。
 */
struct task_group *sched_create_group(struct task_group *parent)
{
	struct task_group *tg;

	tg = kmem_cache_alloc(task_group_cache, GFP_KERNEL | __GFP_ZERO);
	if (!tg)
		return ERR_PTR(-ENOMEM);

	if (!alloc_fair_sched_group(tg, parent))
		goto err;

	if (!alloc_rt_sched_group(tg, parent))
		goto err;

	scx_tg_init(tg);
	alloc_uclamp_sched_group(tg, parent);

	return tg;

err:
	sched_free_group(tg);
	return ERR_PTR(-ENOMEM);
}

/*
 * sched_online_group - 将新任务组发布到全局列表和父子层级，
 * 并上线公平调度资源。
 */
void sched_online_group(struct task_group *tg, struct task_group *parent)
{
	unsigned long flags;

	spin_lock_irqsave(&task_group_lock, flags);
	list_add_tail_rcu(&tg->list, &task_groups);

	/* Root should already exist: */
	WARN_ON(!parent);

	tg->parent = parent;
	INIT_LIST_HEAD(&tg->children);
	list_add_rcu(&tg->siblings, &parent->children);
	spin_unlock_irqrestore(&task_group_lock, flags);

	online_fair_sched_group(tg);
}

/* RCU callback to free various structures associated with a task group */
/*
 * sched_unregister_group_rcu - 等待现有 RCU 读者退出后，
 * 启动任务组调度资源注销。
 */
static void sched_unregister_group_rcu(struct rcu_head *rhp)
{
	/* Now it should be safe to free those cfs_rqs: */
	sched_unregister_group(container_of(rhp, struct task_group, rcu));
}

/*
 * sched_destroy_group - 通过 RCU 延迟注销并最终销毁任务组，
 * 避免并发读取已释放的队列。
 */
void sched_destroy_group(struct task_group *tg)
{
	/* Wait for possible concurrent references to cfs_rqs complete: */
	call_rcu(&tg->rcu, sched_unregister_group_rcu);
}

/*
 * sched_release_group - 从全局任务组列表和父组子链表摘除任务组，
 * 阻止新的层级遍历引用。
 */
void sched_release_group(struct task_group *tg)
{
	unsigned long flags;

	/*
	 * Unlink first, to avoid walk_tg_tree_from() from finding us (via
	 * sched_cfs_period_timer()).
	 *
	 * For this to be effective, we have to wait for all pending users of
	 * this task group to leave their RCU critical section to ensure no new
	 * user will see our dying task group any more. Specifically ensure
	 * that tg_unthrottle_up() won't add decayed cfs_rq's to it.
	 *
	 * We therefore defer calling unregister_fair_sched_group() to
	 * sched_unregister_group() which is guarantied to get called only after the
	 * current RCU grace period has expired.
	 */
	spin_lock_irqsave(&task_group_lock, flags);
	list_del_rcu(&tg->list);
	list_del_rcu(&tg->siblings);
	spin_unlock_irqrestore(&task_group_lock, flags);
}

/*
 * sched_change_group - 依据 cgroup 和自动分组重算任务所属调度组，
 * 并更新组内队列关联。
 * 调用者须持有 task_rq_lock 所保护的任务和队列锁。
 */
static void sched_change_group(struct task_struct *tsk)
{
	struct task_group *tg;

	/*
	 * All callers are synchronized by task_rq_lock(); we do not use RCU
	 * which is pointless here. Thus, we pass "true" to task_css_check()
	 * to prevent lockdep warnings.
	 */
	tg = container_of(task_css_check(tsk, cpu_cgrp_id, true),
			  struct task_group, css);
	tg = autogroup_task_group(tsk, tg);
	tsk->sched_task_group = tg;

	/*
	 * cgroup 决定资源归属，autogroup 可能再映射到自动调度组；
	 * task_change_group/set_task_rq 则把这个归属落实到当前 CPU 的
	 * 类内队列及父实体。只改 sched_task_group 指针会留下旧队列关联。
	 */
#ifdef CONFIG_FAIR_GROUP_SCHED
	if (tsk->sched_class->task_change_group)
		tsk->sched_class->task_change_group(tsk);
	else
#endif
		set_task_rq(tsk, task_cpu(tsk));
}

/*
 * Change task's runqueue when it moves between groups.
 *
 * The caller of this function should have put the task in its new group by
 * now. This function just updates tsk->se.cfs_rq and tsk->se.parent to reflect
 * its new group.
 */
/*
 * sched_move_task - 在任务跨组后重建调度实体的组归属，
 * 恢复队列状态并按需重新调度。
 */
void sched_move_task(struct task_struct *tsk, bool for_autogroup)
{
	unsigned int queue_flags =
		DEQUEUE_SAVE | DEQUEUE_MOVE | DEQUEUE_NOCLOCK;
	bool resched = false;
	struct rq *rq;

	CLASS(task_rq_lock, rq_guard)(tsk);
	rq = rq_guard.rq;

	update_rq_clock(rq);

	/*
	 * 同一个 CPU 上换组也要出旧队列、改 se.parent/cfs_rq、入新队列；
	 * 不必改变 task_cpu。若是当前 donor，更新后请求重新选择，
	 * 让新组的权重/带宽立即参与调度决策。
	 */
	scoped_guard (sched_change, tsk, queue_flags) {
		sched_change_group(tsk);
		if (!for_autogroup)
			scx_cgroup_move_task(tsk);
		if (scope->running)
			resched = true;
	}

	if (resched)
		resched_curr(rq);
}

/*
 * cpu_cgroup_css_alloc - 为 CPU cgroup 分配调度任务组，顶层 cgroup 复用根任务组。
 */
static struct cgroup_subsys_state *
cpu_cgroup_css_alloc(struct cgroup_subsys_state *parent_css)
{
	struct task_group *parent = css_tg(parent_css);
	struct task_group *tg;

	if (!parent) {
		/* This is early initialization for the top cgroup */
		return &root_task_group.css;
	}

	tg = sched_create_group(parent);
	if (IS_ERR(tg))
		return ERR_PTR(-ENOMEM);

	return &tg->css;
}

/* Expose task group only after completing cgroup initialization */
/*
 * cpu_cgroup_css_online - 上线 CPU cgroup 的扩展调度和任务组资源，
 * 并传播有效钳制值。
 */
static int cpu_cgroup_css_online(struct cgroup_subsys_state *css)
{
	struct task_group *tg = css_tg(css);
	struct task_group *parent = css_tg(css->parent);
	int ret;

	ret = scx_tg_online(tg);
	if (ret)
		return ret;

	if (parent)
		sched_online_group(tg, parent);

#ifdef CONFIG_UCLAMP_TASK_GROUP
	/* Propagate the effective uclamp value for the new group */
	guard(mutex)(&uclamp_mutex);
	guard(rcu)();
	cpu_util_update_eff(css);
#endif

	return 0;
}

/*
 * cpu_cgroup_css_offline - CPU cgroup 下线时通知 sched_ext 停用对应任务组。
 */
static void cpu_cgroup_css_offline(struct cgroup_subsys_state *css)
{
	struct task_group *tg = css_tg(css);

	scx_tg_offline(tg);
}

/*
 * cpu_cgroup_css_released - CPU cgroup 被释放时将任务组从调度器可见层级中摘除。
 */
static void cpu_cgroup_css_released(struct cgroup_subsys_state *css)
{
	struct task_group *tg = css_tg(css);

	sched_release_group(tg);
}

/*
 * cpu_cgroup_css_free - 利用 cgroup 生命周期提供的 RCU 宽限期，
 * 注销并延迟释放任务组。
 */
static void cpu_cgroup_css_free(struct cgroup_subsys_state *css)
{
	struct task_group *tg = css_tg(css);

	/*
	 * Relies on the RCU grace period between css_released() and this.
	 */
	sched_unregister_group(tg);
}

/*
 * cpu_cgroup_can_attach - 在任务迁入 CPU cgroup 前
 * 校验实时组限制及 sched_ext 的迁入许可。
 */
static int cpu_cgroup_can_attach(struct cgroup_taskset *tset)
{
#ifdef CONFIG_RT_GROUP_SCHED
	struct task_struct *task;
	struct cgroup_subsys_state *css;

	if (!rt_group_sched_enabled())
		goto scx_check;

	cgroup_taskset_for_each(task, css, tset) {
		if (!sched_rt_can_attach(css_tg(css), task))
			return -EINVAL;
	}
scx_check:
#endif /* CONFIG_RT_GROUP_SCHED */
	return scx_cgroup_can_attach(tset);
}

/*
 * cpu_cgroup_attach - 任务集迁入完成后，
 * 逐个同步任务的调度组和运行队列关联。
 */
static void cpu_cgroup_attach(struct cgroup_taskset *tset)
{
	struct task_struct *task;
	struct cgroup_subsys_state *css;

	cgroup_taskset_for_each(task, css, tset)
		sched_move_task(task, false);
}

/*
 * cpu_cgroup_cancel_attach - 撤销任务集迁入时通知 sched_ext 回滚相关准备。
 */
static void cpu_cgroup_cancel_attach(struct cgroup_taskset *tset)
{
	scx_cgroup_cancel_attach(tset);
}

#ifdef CONFIG_UCLAMP_TASK_GROUP
/*
 * cpu_util_update_eff - 从指定 cgroup 向下传播最严格的利用率钳制，
 * 并刷新受影响任务。
 * 调用者须持有 uclamp_mutex 和 RCU 读锁。
 */
static void cpu_util_update_eff(struct cgroup_subsys_state *css)
{
	struct cgroup_subsys_state *top_css = css;
	struct uclamp_se *uc_parent = NULL;
	struct uclamp_se *uc_se = NULL;
	unsigned int eff[UCLAMP_CNT];
	enum uclamp_id clamp_id;
	unsigned int clamps;

	lockdep_assert_held(&uclamp_mutex);
	WARN_ON_ONCE(!rcu_read_lock_held());

	css_for_each_descendant_pre(css, top_css) {
		/*
		 * 前序遍历先更新父再更新子。这里两个维度均由父的有效值
		 * 作为上界：子组不能请求比父组允许的更高保护或更高上限，
		 * 最后再把 min 截到 max，保持有效区间不颠倒。
		 */
		uc_parent = css_tg(css)->parent
			? css_tg(css)->parent->uclamp : NULL;

		for_each_clamp_id(clamp_id) {
			/* Assume effective clamps matches requested clamps */
			eff[clamp_id] = css_tg(css)->uclamp_req[clamp_id].value;
			/* Cap effective clamps with parent's effective clamps */
			if (uc_parent &&
			    eff[clamp_id] > uc_parent[clamp_id].value) {
				eff[clamp_id] = uc_parent[clamp_id].value;
			}
		}
		/* Ensure protection is always capped by limit */
		eff[UCLAMP_MIN] = min(eff[UCLAMP_MIN], eff[UCLAMP_MAX]);

		/* Propagate most restrictive effective clamps */
		clamps = 0x0;
		uc_se = css_tg(css)->uclamp;
		for_each_clamp_id(clamp_id) {
			if (eff[clamp_id] == uc_se[clamp_id].value)
				continue;
			uc_se[clamp_id].value = eff[clamp_id];
			uc_se[clamp_id].bucket_id = uclamp_bucket_id(eff[clamp_id]);
			clamps |= (0x1 << clamp_id);
		}
		if (!clamps) {
			/* 有效约束未变，后代输入也未变，可跳过子树。 */
			css = css_rightmost_descendant(css);
			continue;
		}

		/* Immediately update descendants RUNNABLE tasks */
		uclamp_update_active_tasks(css);
	}
}

/*
 * Integer 10^N with a given N exponent by casting to integer the literal "1eN"
 * C expression. Since there is no way to convert a macro argument (N) into a
 * character constant, use two levels of macros.
 */
#define _POW10(exp) ((unsigned int)1e##exp)
#define POW10(exp) _POW10(exp)

struct uclamp_request {
#define UCLAMP_PERCENT_SHIFT	2
#define UCLAMP_PERCENT_SCALE	(100 * POW10(UCLAMP_PERCENT_SHIFT))
	s64 percent;
	u64 util;
	int ret;
};

/*
 * capacity_from_percent - 将百分比或 max 文本解析为利用率容量，
 * 并在结果中携带错误码。
 */
static inline struct uclamp_request
capacity_from_percent(char *buf)
{
	struct uclamp_request req = {
		.percent = UCLAMP_PERCENT_SCALE,
		.util = SCHED_CAPACITY_SCALE,
		.ret = 0,
	};

	buf = strim(buf);
	if (strcmp(buf, "max")) {
		req.ret = cgroup_parse_float(buf, UCLAMP_PERCENT_SHIFT,
					     &req.percent);
		if (req.ret)
			return req;
		if ((u64)req.percent > UCLAMP_PERCENT_SCALE) {
			req.ret = -ERANGE;
			return req;
		}

		req.util = req.percent << SCHED_CAPACITY_SHIFT;
		req.util = DIV_ROUND_CLOSEST_ULL(req.util, UCLAMP_PERCENT_SCALE);
	}

	return req;
}

/*
 * cpu_uclamp_write - 写入 cgroup 指定类型的钳制请求，
 * 保留原始百分比并更新子树有效值。
 * 成功返回写入字节数，失败返回负错误码。
 */
static ssize_t cpu_uclamp_write(struct kernfs_open_file *of, char *buf,
				size_t nbytes, loff_t off,
				enum uclamp_id clamp_id)
{
	struct uclamp_request req;
	struct task_group *tg;

	req = capacity_from_percent(buf);
	if (req.ret)
		return req.ret;

	sched_uclamp_enable();

	guard(mutex)(&uclamp_mutex);
	guard(rcu)();

	tg = css_tg(of_css(of));
	if (tg->uclamp_req[clamp_id].value != req.util)
		uclamp_se_set(&tg->uclamp_req[clamp_id], req.util, false);

	/*
	 * Because of not recoverable conversion rounding we keep track of the
	 * exact requested value
	 */
	tg->uclamp_pct[clamp_id] = req.percent;

	/* Update effective clamps to track the most restrictive value */
	cpu_util_update_eff(of_css(of));

	return nbytes;
}

/*
 * cpu_uclamp_min_write - 处理 cpu.uclamp.min 写入，设置任务组最小利用率请求。
 */
static ssize_t cpu_uclamp_min_write(struct kernfs_open_file *of,
				    char *buf, size_t nbytes,
				    loff_t off)
{
	return cpu_uclamp_write(of, buf, nbytes, off, UCLAMP_MIN);
}

/*
 * cpu_uclamp_max_write - 处理 cpu.uclamp.max 写入，设置任务组最大利用率请求。
 */
static ssize_t cpu_uclamp_max_write(struct kernfs_open_file *of,
				    char *buf, size_t nbytes,
				    loff_t off)
{
	return cpu_uclamp_write(of, buf, nbytes, off, UCLAMP_MAX);
}

/*
 * cpu_uclamp_print - 以 max 或保留两位小数的百分比格式
 * 输出任务组请求的钳制值。
 */
static inline void cpu_uclamp_print(struct seq_file *sf,
				    enum uclamp_id clamp_id)
{
	struct task_group *tg;
	u64 util_clamp;
	u64 percent;
	u32 rem;

	scoped_guard (rcu) {
		tg = css_tg(seq_css(sf));
		util_clamp = tg->uclamp_req[clamp_id].value;
	}

	if (util_clamp == SCHED_CAPACITY_SCALE) {
		seq_puts(sf, "max\n");
		return;
	}

	percent = tg->uclamp_pct[clamp_id];
	percent = div_u64_rem(percent, POW10(UCLAMP_PERCENT_SHIFT), &rem);
	seq_printf(sf, "%llu.%0*u\n", percent, UCLAMP_PERCENT_SHIFT, rem);
}

/*
 * cpu_uclamp_min_show - 输出 cpu.uclamp.min 的当前请求值。
 */
static int cpu_uclamp_min_show(struct seq_file *sf, void *v)
{
	cpu_uclamp_print(sf, UCLAMP_MIN);
	return 0;
}

/*
 * cpu_uclamp_max_show - 输出 cpu.uclamp.max 的当前请求值。
 */
static int cpu_uclamp_max_show(struct seq_file *sf, void *v)
{
	cpu_uclamp_print(sf, UCLAMP_MAX);
	return 0;
}
#endif /* CONFIG_UCLAMP_TASK_GROUP */

#ifdef CONFIG_GROUP_SCHED_WEIGHT
/*
 * tg_weight - 读取任务组的统一调度权重，
 * 兼容公平调度及仅扩展调度的配置。
 */
static unsigned long tg_weight(struct task_group *tg)
{
#ifdef CONFIG_FAIR_GROUP_SCHED
	return scale_load_down(tg->shares);
#else
	return sched_weight_from_cgroup(tg->scx.weight);
#endif
}

/*
 * cpu_shares_write_u64 - 设置旧版 cgroup CPU shares，并在成功后同步 sched_ext 权重。
 */
static int cpu_shares_write_u64(struct cgroup_subsys_state *css,
				struct cftype *cftype, u64 shareval)
{
	int ret;

	if (shareval > scale_load_down(ULONG_MAX))
		shareval = MAX_SHARES;
	ret = sched_group_set_shares(css_tg(css), scale_load(shareval));
	if (!ret)
		scx_group_set_weight(css_tg(css),
				     sched_weight_to_cgroup(shareval));
	return ret;
}

/*
 * cpu_shares_read_u64 - 读取旧版 CPU cgroup 的任务组 shares 权重。
 */
static u64 cpu_shares_read_u64(struct cgroup_subsys_state *css,
			       struct cftype *cft)
{
	return tg_weight(css_tg(css));
}
#endif /* CONFIG_GROUP_SCHED_WEIGHT */

#ifdef CONFIG_CFS_BANDWIDTH
static DEFINE_MUTEX(cfs_constraints_mutex);

static int __cfs_schedulable(struct task_group *tg, u64 period, u64 runtime);

/*
 * tg_set_cfs_bandwidth - 应用以微秒传入的公平组周期、配额及突发预算，
 * 并刷新各 CPU 队列。
 * 校验层级约束后重新补充预算、调整定时器并解除旧的限流状态。
 */
static int tg_set_cfs_bandwidth(struct task_group *tg,
				u64 period_us, u64 quota_us, u64 burst_us)
{
	int i, ret = 0, runtime_enabled, runtime_was_enabled;
	struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;
	u64 period, quota, burst;

	/*
	 * 用户接口使用微秒，内部预算使用纳秒；RUNTIME_INF 必须原样
	 * 保留，不能参与乘法。quota 是一个组跨 CPU 共享的周期预算，
	 * 各 CPU 的 cfs_rq 再从组预算取得本地运行额度。
	 */
	period = (u64)period_us * NSEC_PER_USEC;

	if (quota_us == RUNTIME_INF)
		quota = RUNTIME_INF;
	else
		quota = (u64)quota_us * NSEC_PER_USEC;

	burst = (u64)burst_us * NSEC_PER_USEC;

	/*
	 * Prevent race between setting of cfs_rq->runtime_enabled and
	 * unthrottle_offline_cfs_rqs().
	 */
	guard(cpus_read_lock)();
	guard(mutex)(&cfs_constraints_mutex);

	/*
	 * 锁分工：cpus_read_lock 稳住 CPU 热插拔，约束互斥锁串行化
	 * 组树带宽修改，cfs_b->lock 保护共享预算，rq 锁保护本地额度。
	 * 下方先释放 cfs_b->lock 再逐个获取 rq 锁，不能把两者倒序嵌套；
	 * 运行时分配额度的路径会在 rq 锁内获取带宽锁。
	 */
	/* 先用候选值验证整棵组树，失败时不发布新的带宽配置。 */
	ret = __cfs_schedulable(tg, period, quota);
	if (ret)
		return ret;

	runtime_enabled = quota != RUNTIME_INF;
	runtime_was_enabled = cfs_b->quota != RUNTIME_INF;
	/*
	 * If we need to toggle cfs_bandwidth_used, off->on must occur
	 * before making related changes, and on->off must occur afterwards
	 */
	if (runtime_enabled && !runtime_was_enabled)
		cfs_bandwidth_usage_inc();

	scoped_guard (raw_spinlock_irq, &cfs_b->lock) {
		cfs_b->period = ns_to_ktime(period);
		cfs_b->quota = quota;
		cfs_b->burst = burst;

		__refill_cfs_bandwidth_runtime(cfs_b);

		/*
		 * Restart the period timer (if active) to handle new
		 * period expiry:
		 */
		if (runtime_enabled)
			start_cfs_bandwidth(cfs_b);
	}

	for_each_online_cpu(i) {
		struct cfs_rq *cfs_rq = tg->cfs_rq[i];
		struct rq *rq = cfs_rq->rq;

		/*
		 * 组级预算已更新，再逐 CPU 刷新本地执行许可。不能保留旧的
		 * 负债和限流状态，否则放宽配额后可能无谓地等待旧周期。
		 */
		guard(rq_lock_irq)(rq);
		cfs_rq->runtime_enabled = runtime_enabled;
		cfs_rq->runtime_remaining = 1;

		if (cfs_rq->throttled)
			unthrottle_cfs_rq(cfs_rq);
	}

	if (runtime_was_enabled && !runtime_enabled)
		cfs_bandwidth_usage_dec();

	return 0;
}

/*
 * tg_get_cfs_period - 将公平调度任务组的带宽周期转换为微秒并返回。
 */
static u64 tg_get_cfs_period(struct task_group *tg)
{
	u64 cfs_period_us;

	cfs_period_us = ktime_to_ns(tg->cfs_bandwidth.period);
	do_div(cfs_period_us, NSEC_PER_USEC);

	return cfs_period_us;
}

/*
 * tg_get_cfs_quota - 返回任务组的微秒配额，不限额时保持 RUNTIME_INF 标记。
 */
static u64 tg_get_cfs_quota(struct task_group *tg)
{
	u64 quota_us;

	if (tg->cfs_bandwidth.quota == RUNTIME_INF)
		return RUNTIME_INF;

	quota_us = tg->cfs_bandwidth.quota;
	do_div(quota_us, NSEC_PER_USEC);

	return quota_us;
}

/*
 * tg_get_cfs_burst - 返回公平调度任务组允许积累的突发运行预算，
 * 单位为微秒。
 */
static u64 tg_get_cfs_burst(struct task_group *tg)
{
	u64 burst_us;

	burst_us = tg->cfs_bandwidth.burst;
	do_div(burst_us, NSEC_PER_USEC);

	return burst_us;
}

struct cfs_schedulable_data {
	struct task_group *tg;
	u64 period, quota;
};

/*
 * normalize group quota/period to be quota/max_period
 * note: units are usecs
 */
/*
 * normalize_cfs_quota - 将任务组当前或候选配额归一化为定点带宽比例，
 * 保留无限额语义。
 */
static u64 normalize_cfs_quota(struct task_group *tg,
			       struct cfs_schedulable_data *d)
{
	u64 quota, period;

	if (tg == d->tg) {
		period = d->period;
		quota = d->quota;
	} else {
		period = tg_get_cfs_period(tg);
		quota = tg_get_cfs_quota(tg);
	}

	/* note: these should typically be equivalent */
	if (quota == RUNTIME_INF || quota == -1)
		return RUNTIME_INF;

	return to_ratio(period, quota);
}

/*
 * tg_cfs_schedulable_down - 遍历任务组时计算层级有效配额，
 * 并检查旧版 cgroup 的父组约束。
 */
static int tg_cfs_schedulable_down(struct task_group *tg, void *data)
{
	struct cfs_schedulable_data *d = data;
	struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;
	s64 quota = 0, parent_quota = -1;

	/*
	 * 比较的是 quota/period 的归一化比例，不是配额的绝对微秒数。
	 * 此检查没有要求所有子组配额之和小于父组：子组可以超额配置，
	 * 同时繁忙时仍共同受父组预算限制。v1 对单个子组的显式上限
	 * 超过父组时报错，v2 则取父子有效上限的较小者。
	 */
	if (!tg->parent) {
		quota = RUNTIME_INF;
	} else {
		struct cfs_bandwidth *parent_b = &tg->parent->cfs_bandwidth;

		quota = normalize_cfs_quota(tg, d);
		parent_quota = parent_b->hierarchical_quota;

		/*
		 * Ensure max(child_quota) <= parent_quota.  On cgroup2,
		 * always take the non-RUNTIME_INF min.  On cgroup1, only
		 * inherit when no limit is set. In both cases this is used
		 * by the scheduler to determine if a given CFS task has a
		 * bandwidth constraint at some higher level.
		 */
		if (cgroup_subsys_on_dfl(cpu_cgrp_subsys)) {
			if (quota == RUNTIME_INF)
				quota = parent_quota;
			else if (parent_quota != RUNTIME_INF)
				quota = min(quota, parent_quota);
		} else {
			if (quota == RUNTIME_INF)
				quota = parent_quota;
			else if (parent_quota != RUNTIME_INF && quota > parent_quota)
				return -EINVAL;
		}
	}
	cfs_b->hierarchical_quota = quota;

	return 0;
}

/*
 * __cfs_schedulable - 用候选周期和配额遍历任务组树，
 * 校验并更新层级带宽可调度性。
 * 输入单位为纳秒，成功返回 0，违反层级限制时返回错误码。
 */
static int __cfs_schedulable(struct task_group *tg, u64 period, u64 quota)
{
	struct cfs_schedulable_data data = {
		.tg = tg,
		.period = period,
		.quota = quota,
	};

	if (quota != RUNTIME_INF) {
		do_div(data.period, NSEC_PER_USEC);
		do_div(data.quota, NSEC_PER_USEC);
	}

	guard(rcu)();
	return walk_tg_tree(tg_cfs_schedulable_down, tg_nop, &data);
}

/*
 * cpu_cfs_stat_show - 输出旧版 cgroup 的公平带宽周期、限流、等待及突发统计。
 */
static int cpu_cfs_stat_show(struct seq_file *sf, void *v)
{
	struct task_group *tg = css_tg(seq_css(sf));
	struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;

	seq_printf(sf, "nr_periods %d\n", cfs_b->nr_periods);
	seq_printf(sf, "nr_throttled %d\n", cfs_b->nr_throttled);
	seq_printf(sf, "throttled_time %llu\n", cfs_b->throttled_time);

	if (schedstat_enabled() && tg != &root_task_group) {
		struct sched_statistics *stats;
		u64 ws = 0;
		int i;

		for_each_possible_cpu(i) {
			stats = __schedstats_from_se(tg->se[i]);
			ws += schedstat_val(stats->wait_sum);
		}

		seq_printf(sf, "wait_sum %llu\n", ws);
	}

	seq_printf(sf, "nr_bursts %d\n", cfs_b->nr_burst);
	seq_printf(sf, "burst_time %llu\n", cfs_b->burst_time);

	return 0;
}

/*
 * throttled_time_self - 汇总任务组各 CPU 队列自身累计的限流时间，
 * 单位为纳秒。
 */
static u64 throttled_time_self(struct task_group *tg)
{
	int i;
	u64 total = 0;

	for_each_possible_cpu(i) {
		total += READ_ONCE(tg->cfs_rq[i]->throttled_clock_self_time);
	}

	return total;
}

/*
 * cpu_cfs_local_stat_show - 输出旧版 CPU cgroup 的本组限流时间统计。
 */
static int cpu_cfs_local_stat_show(struct seq_file *sf, void *v)
{
	struct task_group *tg = css_tg(seq_css(sf));

	seq_printf(sf, "throttled_time %llu\n", throttled_time_self(tg));

	return 0;
}
#endif /* CONFIG_CFS_BANDWIDTH */

#ifdef CONFIG_GROUP_SCHED_BANDWIDTH
const u64 max_bw_quota_period_us = 1 * USEC_PER_SEC; /* 1s */
static const u64 min_bw_quota_period_us = 1 * USEC_PER_MSEC; /* 1ms */
/* More than 203 days if BW_SHIFT equals 20. */
static const u64 max_bw_runtime_us = MAX_BW;

/*
 * tg_bandwidth - 从公平或扩展调度组中读取周期、配额和突发预算，
 * 输出单位为微秒。
 * 输出指针为 NULL 的字段不读取。
 */
static void tg_bandwidth(struct task_group *tg,
			 u64 *period_us_p, u64 *quota_us_p, u64 *burst_us_p)
{
#ifdef CONFIG_CFS_BANDWIDTH
	if (period_us_p)
		*period_us_p = tg_get_cfs_period(tg);
	if (quota_us_p)
		*quota_us_p = tg_get_cfs_quota(tg);
	if (burst_us_p)
		*burst_us_p = tg_get_cfs_burst(tg);
#else /* !CONFIG_CFS_BANDWIDTH */
	if (period_us_p)
		*period_us_p = tg->scx.bw_period_us;
	if (quota_us_p)
		*quota_us_p = tg->scx.bw_quota_us;
	if (burst_us_p)
		*burst_us_p = tg->scx.bw_burst_us;
#endif /* CONFIG_CFS_BANDWIDTH */
}

/*
 * cpu_period_read_u64 - 返回 CPU cgroup 的带宽控制周期，单位为微秒。
 */
static u64 cpu_period_read_u64(struct cgroup_subsys_state *css,
			       struct cftype *cft)
{
	u64 period_us;

	tg_bandwidth(css_tg(css), &period_us, NULL, NULL);
	return period_us;
}

/*
 * tg_set_bandwidth - 校验任务组带宽参数及溢出边界，
 * 再依次应用到公平和扩展调度器。
 * 输入单位为微秒，成功返回 0，非法参数或应用失败时返回负错误码。
 */
static int tg_set_bandwidth(struct task_group *tg,
			    u64 period_us, u64 quota_us, u64 burst_us)
{
	const u64 max_usec = U64_MAX / NSEC_PER_USEC;
	int ret = 0;

	/*
	 * 带宽、权重和 uclamp 是三种不同控制：quota/period 限累计时间，
	 * weight 决定有竞争时的相对份额，uclamp 影响性能需求信号。
	 * 例如 quota=200000us、period=100000us 是组总计两个 CPU 的
	 * 平均预算，允许任务在多个 CPU 并行，并非固定单 CPU 的时间片。
	 */
	if (tg == &root_task_group)
		return -EINVAL;

	/* Values should survive translation to nsec */
	if (period_us > max_usec ||
	    (quota_us != RUNTIME_INF && quota_us > max_usec) ||
	    burst_us > max_usec)
		return -EINVAL;

	/*
	 * Ensure we have some amount of bandwidth every period. This is to
	 * prevent reaching a state of large arrears when throttled via
	 * entity_tick() resulting in prolonged exit starvation.
	 */
	if (quota_us < min_bw_quota_period_us ||
	    period_us < min_bw_quota_period_us)
		return -EINVAL;

	/*
	 * Likewise, bound things on the other side by preventing insane quota
	 * periods.  This also allows us to normalize in computing quota
	 * feasibility.
	 */
	if (period_us > max_bw_quota_period_us)
		return -EINVAL;

	/*
	 * Bound quota to defend quota against overflow during bandwidth shift.
	 */
	if (quota_us != RUNTIME_INF && quota_us > max_bw_runtime_us)
		return -EINVAL;

	if (quota_us != RUNTIME_INF && (burst_us > quota_us ||
					burst_us + quota_us > max_bw_runtime_us))
		return -EINVAL;

#ifdef CONFIG_CFS_BANDWIDTH
	ret = tg_set_cfs_bandwidth(tg, period_us, quota_us, burst_us);
#endif /* CONFIG_CFS_BANDWIDTH */
	if (!ret)
		scx_group_set_bandwidth(tg, period_us, quota_us, burst_us);
	return ret;
}

/*
 * cpu_quota_read_s64 - 读取任务组的微秒运行配额，以 -1 表示不限制。
 */
static s64 cpu_quota_read_s64(struct cgroup_subsys_state *css,
			      struct cftype *cft)
{
	u64 quota_us;

	tg_bandwidth(css_tg(css), NULL, &quota_us, NULL);
	return quota_us;	/* (s64)RUNTIME_INF becomes -1 */
}

/*
 * cpu_burst_read_u64 - 读取任务组允许的突发运行预算，单位为微秒。
 */
static u64 cpu_burst_read_u64(struct cgroup_subsys_state *css,
			      struct cftype *cft)
{
	u64 burst_us;

	tg_bandwidth(css_tg(css), NULL, NULL, &burst_us);
	return burst_us;
}

/*
 * cpu_period_write_u64 - 保持现有配额和突发预算，更新任务组的带宽周期。
 */
static int cpu_period_write_u64(struct cgroup_subsys_state *css,
				struct cftype *cftype, u64 period_us)
{
	struct task_group *tg = css_tg(css);
	u64 quota_us, burst_us;

	tg_bandwidth(tg, NULL, &quota_us, &burst_us);
	return tg_set_bandwidth(tg, period_us, quota_us, burst_us);
}

/*
 * cpu_quota_write_s64 - 更新任务组运行配额，
 * 负值转换为无限额，并保留周期及突发预算。
 */
static int cpu_quota_write_s64(struct cgroup_subsys_state *css,
			       struct cftype *cftype, s64 quota_us)
{
	struct task_group *tg = css_tg(css);
	u64 period_us, burst_us;

	if (quota_us < 0)
		quota_us = RUNTIME_INF;

	tg_bandwidth(tg, &period_us, NULL, &burst_us);
	return tg_set_bandwidth(tg, period_us, quota_us, burst_us);
}

/*
 * cpu_burst_write_u64 - 保持现有周期和配额，更新任务组的突发运行预算。
 */
static int cpu_burst_write_u64(struct cgroup_subsys_state *css,
			       struct cftype *cftype, u64 burst_us)
{
	struct task_group *tg = css_tg(css);
	u64 period_us, quota_us;

	tg_bandwidth(tg, &period_us, &quota_us, NULL);
	return tg_set_bandwidth(tg, period_us, quota_us, burst_us);
}
#endif /* CONFIG_GROUP_SCHED_BANDWIDTH */

#ifdef CONFIG_RT_GROUP_SCHED
/*
 * cpu_rt_runtime_write - 设置任务组实时调度的运行预算，单位为微秒。
 */
static int cpu_rt_runtime_write(struct cgroup_subsys_state *css,
				struct cftype *cft, s64 val)
{
	return sched_group_set_rt_runtime(css_tg(css), val);
}

/*
 * cpu_rt_runtime_read - 读取任务组实时调度的微秒预算，负值表示无限额。
 */
static s64 cpu_rt_runtime_read(struct cgroup_subsys_state *css,
			       struct cftype *cft)
{
	return sched_group_rt_runtime(css_tg(css));
}

/*
 * cpu_rt_period_write_uint - 设置任务组实时调度预算的补充周期，单位为微秒。
 */
static int cpu_rt_period_write_uint(struct cgroup_subsys_state *css,
				    struct cftype *cftype, u64 rt_period_us)
{
	return sched_group_set_rt_period(css_tg(css), rt_period_us);
}

/*
 * cpu_rt_period_read_uint - 读取任务组实时调度的带宽周期，单位为微秒。
 */
static u64 cpu_rt_period_read_uint(struct cgroup_subsys_state *css,
				   struct cftype *cft)
{
	return sched_group_rt_period(css_tg(css));
}
#endif /* CONFIG_RT_GROUP_SCHED */

#ifdef CONFIG_GROUP_SCHED_WEIGHT
/*
 * cpu_idle_read_s64 - 读取 CPU cgroup 的空闲调度组属性。
 */
static s64 cpu_idle_read_s64(struct cgroup_subsys_state *css,
			       struct cftype *cft)
{
	return css_tg(css)->idle;
}

/*
 * cpu_idle_write_s64 - 设置任务组空闲调度属性，成功后同步至 sched_ext。
 */
static int cpu_idle_write_s64(struct cgroup_subsys_state *css,
				struct cftype *cft, s64 idle)
{
	int ret;

	ret = sched_group_set_idle(css_tg(css), idle);
	if (!ret)
		scx_group_set_idle(css_tg(css), idle);
	return ret;
}
#endif /* CONFIG_GROUP_SCHED_WEIGHT */

static struct cftype cpu_legacy_files[] = {
#ifdef CONFIG_GROUP_SCHED_WEIGHT
	{
		.name = "shares",
		.read_u64 = cpu_shares_read_u64,
		.write_u64 = cpu_shares_write_u64,
	},
	{
		.name = "idle",
		.read_s64 = cpu_idle_read_s64,
		.write_s64 = cpu_idle_write_s64,
	},
#endif
#ifdef CONFIG_GROUP_SCHED_BANDWIDTH
	{
		.name = "cfs_period_us",
		.read_u64 = cpu_period_read_u64,
		.write_u64 = cpu_period_write_u64,
	},
	{
		.name = "cfs_quota_us",
		.read_s64 = cpu_quota_read_s64,
		.write_s64 = cpu_quota_write_s64,
	},
	{
		.name = "cfs_burst_us",
		.read_u64 = cpu_burst_read_u64,
		.write_u64 = cpu_burst_write_u64,
	},
#endif
#ifdef CONFIG_CFS_BANDWIDTH
	{
		.name = "stat",
		.seq_show = cpu_cfs_stat_show,
	},
	{
		.name = "stat.local",
		.seq_show = cpu_cfs_local_stat_show,
	},
#endif
#ifdef CONFIG_UCLAMP_TASK_GROUP
	{
		.name = "uclamp.min",
		.flags = CFTYPE_NOT_ON_ROOT,
		.seq_show = cpu_uclamp_min_show,
		.write = cpu_uclamp_min_write,
	},
	{
		.name = "uclamp.max",
		.flags = CFTYPE_NOT_ON_ROOT,
		.seq_show = cpu_uclamp_max_show,
		.write = cpu_uclamp_max_write,
	},
#endif
	{ }	/* Terminate */
};

#ifdef CONFIG_RT_GROUP_SCHED
static struct cftype rt_group_files[] = {
	{
		.name = "rt_runtime_us",
		.read_s64 = cpu_rt_runtime_read,
		.write_s64 = cpu_rt_runtime_write,
	},
	{
		.name = "rt_period_us",
		.read_u64 = cpu_rt_period_read_uint,
		.write_u64 = cpu_rt_period_write_uint,
	},
	{ }	/* Terminate */
};

# ifdef CONFIG_RT_GROUP_SCHED_DEFAULT_DISABLED
DEFINE_STATIC_KEY_FALSE(rt_group_sched);
# else
DEFINE_STATIC_KEY_TRUE(rt_group_sched);
# endif

/*
 * setup_rt_group_sched - 解析 rt_group_sched 启动参数，
 * 按 0 或 1 切换实时组调度静态分支。
 */
static int __init setup_rt_group_sched(char *str)
{
	long val;

	if (kstrtol(str, 0, &val) || val < 0 || val > 1) {
		pr_warn("Unable to set rt_group_sched\n");
		return 1;
	}
	if (val)
		static_branch_enable(&rt_group_sched);
	else
		static_branch_disable(&rt_group_sched);

	return 1;
}
__setup("rt_group_sched=", setup_rt_group_sched);

/*
 * cpu_rt_group_init - 实时组调度启用时，
 * 注册旧版 CPU cgroup 的实时带宽控制文件。
 */
static int __init cpu_rt_group_init(void)
{
	if (!rt_group_sched_enabled())
		return 0;

	WARN_ON(cgroup_add_legacy_cftypes(&cpu_cgrp_subsys, rt_group_files));
	return 0;
}
subsys_initcall(cpu_rt_group_init);
#endif /* CONFIG_RT_GROUP_SCHED */

/*
 * cpu_extra_stat_show - 输出 CPU cgroup 的附加带宽统计，时间统一换算为微秒。
 */
static int cpu_extra_stat_show(struct seq_file *sf,
			       struct cgroup_subsys_state *css)
{
#ifdef CONFIG_CFS_BANDWIDTH
	{
		struct task_group *tg = css_tg(css);
		struct cfs_bandwidth *cfs_b = &tg->cfs_bandwidth;
		u64 throttled_usec, burst_usec;

		throttled_usec = cfs_b->throttled_time;
		do_div(throttled_usec, NSEC_PER_USEC);
		burst_usec = cfs_b->burst_time;
		do_div(burst_usec, NSEC_PER_USEC);

		seq_printf(sf, "nr_periods %d\n"
			   "nr_throttled %d\n"
			   "throttled_usec %llu\n"
			   "nr_bursts %d\n"
			   "burst_usec %llu\n",
			   cfs_b->nr_periods, cfs_b->nr_throttled,
			   throttled_usec, cfs_b->nr_burst, burst_usec);
	}
#endif /* CONFIG_CFS_BANDWIDTH */
	return 0;
}

/*
 * cpu_local_stat_show - 输出 CPU cgroup 自身的累计限流时间，单位为微秒。
 */
static int cpu_local_stat_show(struct seq_file *sf,
			       struct cgroup_subsys_state *css)
{
#ifdef CONFIG_CFS_BANDWIDTH
	{
		struct task_group *tg = css_tg(css);
		u64 throttled_self_usec;

		throttled_self_usec = throttled_time_self(tg);
		do_div(throttled_self_usec, NSEC_PER_USEC);

		seq_printf(sf, "throttled_usec %llu\n",
			   throttled_self_usec);
	}
#endif
	return 0;
}

#ifdef CONFIG_GROUP_SCHED_WEIGHT

/*
 * cpu_weight_read_u64 - 将任务组内部权重转换为 cgroup v2 的 CPU weight 并返回。
 */
static u64 cpu_weight_read_u64(struct cgroup_subsys_state *css,
			       struct cftype *cft)
{
	return sched_weight_to_cgroup(tg_weight(css_tg(css)));
}

/*
 * cpu_weight_write_u64 - 校验并设置 cgroup v2 CPU weight，
 * 同步公平及扩展调度权重。
 */
static int cpu_weight_write_u64(struct cgroup_subsys_state *css,
				struct cftype *cft, u64 cgrp_weight)
{
	unsigned long weight;
	int ret;

	if (cgrp_weight < CGROUP_WEIGHT_MIN || cgrp_weight > CGROUP_WEIGHT_MAX)
		return -ERANGE;

	weight = sched_weight_from_cgroup(cgrp_weight);

	ret = sched_group_set_shares(css_tg(css), scale_load(weight));
	if (!ret)
		scx_group_set_weight(css_tg(css), cgrp_weight);
	return ret;
}

/*
 * cpu_weight_nice_read_s64 - 在标准权重表中
 * 寻找最接近任务组权重的 nice 值并返回。
 */
static s64 cpu_weight_nice_read_s64(struct cgroup_subsys_state *css,
				    struct cftype *cft)
{
	unsigned long weight = tg_weight(css_tg(css));
	int last_delta = INT_MAX;
	int prio, delta;

	/* find the closest nice value to the current weight */
	for (prio = 0; prio < ARRAY_SIZE(sched_prio_to_weight); prio++) {
		delta = abs(sched_prio_to_weight[prio] - weight);
		if (delta >= last_delta)
			break;
		last_delta = delta;
	}

	return PRIO_TO_NICE(prio - 1 + MAX_RT_PRIO);
}

/*
 * cpu_weight_nice_write_s64 - 校验 nice 范围并转换为任务组权重，
 * 成功后同步 sched_ext。
 */
static int cpu_weight_nice_write_s64(struct cgroup_subsys_state *css,
				     struct cftype *cft, s64 nice)
{
	unsigned long weight;
	int idx, ret;

	if (nice < MIN_NICE || nice > MAX_NICE)
		return -ERANGE;

	idx = NICE_TO_PRIO(nice) - MAX_RT_PRIO;
	idx = array_index_nospec(idx, 40);
	weight = sched_prio_to_weight[idx];

	ret = sched_group_set_shares(css_tg(css), scale_load(weight));
	if (!ret)
		scx_group_set_weight(css_tg(css),
				     sched_weight_to_cgroup(weight));
	return ret;
}
#endif /* CONFIG_GROUP_SCHED_WEIGHT */

/*
 * cpu_period_quota_print - 按 cpu.max 格式输出配额和周期，负配额显示为 max。
 */
static void __maybe_unused cpu_period_quota_print(struct seq_file *sf,
						  long period, long quota)
{
	if (quota < 0)
		seq_puts(sf, "max");
	else
		seq_printf(sf, "%ld", quota);

	seq_printf(sf, " %ld\n", period);
}

/* caller should put the current value in *@periodp before calling */
/*
 * cpu_period_quota_parse - 解析配额及可选周期文本，
 * 支持 max，省略周期时保留调用者原值。
 * 输出单位为微秒，解析成功返回 0，格式无效返回 -EINVAL。
 */
static int __maybe_unused cpu_period_quota_parse(char *buf, u64 *period_us_p,
						 u64 *quota_us_p)
{
	char tok[21];	/* U64_MAX */

	if (sscanf(buf, "%20s %llu", tok, period_us_p) < 1)
		return -EINVAL;

	if (sscanf(tok, "%llu", quota_us_p) < 1) {
		if (!strcmp(tok, "max"))
			*quota_us_p = RUNTIME_INF;
		else
			return -EINVAL;
	}

	return 0;
}

#ifdef CONFIG_GROUP_SCHED_BANDWIDTH
/*
 * cpu_max_show - 按 cgroup v2 cpu.max 格式输出任务组的当前配额和周期。
 */
static int cpu_max_show(struct seq_file *sf, void *v)
{
	struct task_group *tg = css_tg(seq_css(sf));
	u64 period_us, quota_us;

	tg_bandwidth(tg, &period_us, &quota_us, NULL);
	cpu_period_quota_print(sf, period_us, quota_us);
	return 0;
}

/*
 * cpu_max_write - 解析并应用 cpu.max 的配额和可选周期，保留已有突发预算。
 * 成功返回写入字节数，失败返回负错误码。
 */
static ssize_t cpu_max_write(struct kernfs_open_file *of,
			     char *buf, size_t nbytes, loff_t off)
{
	struct task_group *tg = css_tg(of_css(of));
	u64 period_us, quota_us, burst_us;
	int ret;

	tg_bandwidth(tg, &period_us, NULL, &burst_us);
	ret = cpu_period_quota_parse(buf, &period_us, &quota_us);
	if (!ret)
		ret = tg_set_bandwidth(tg, period_us, quota_us, burst_us);
	return ret ?: nbytes;
}
#endif /* CONFIG_CFS_BANDWIDTH */

/*
 * cgroup v2 文件到上面控制逻辑的映射：weight/weight.nice 调相对权重，
 * max/max.burst 调时间预算，uclamp.* 调容量信号。cftype 只注册接口，
 * 实际参数校验、层级传播和各 CPU 队列更新由相应回调完成。
 */
static struct cftype cpu_files[] = {
#ifdef CONFIG_GROUP_SCHED_WEIGHT
	{
		.name = "weight",
		.flags = CFTYPE_NOT_ON_ROOT,
		.read_u64 = cpu_weight_read_u64,
		.write_u64 = cpu_weight_write_u64,
	},
	{
		.name = "weight.nice",
		.flags = CFTYPE_NOT_ON_ROOT,
		.read_s64 = cpu_weight_nice_read_s64,
		.write_s64 = cpu_weight_nice_write_s64,
	},
	{
		.name = "idle",
		.flags = CFTYPE_NOT_ON_ROOT,
		.read_s64 = cpu_idle_read_s64,
		.write_s64 = cpu_idle_write_s64,
	},
#endif
#ifdef CONFIG_GROUP_SCHED_BANDWIDTH
	{
		.name = "max",
		.flags = CFTYPE_NOT_ON_ROOT,
		.seq_show = cpu_max_show,
		.write = cpu_max_write,
	},
	{
		.name = "max.burst",
		.flags = CFTYPE_NOT_ON_ROOT,
		.read_u64 = cpu_burst_read_u64,
		.write_u64 = cpu_burst_write_u64,
	},
#endif /* CONFIG_CFS_BANDWIDTH */
#ifdef CONFIG_UCLAMP_TASK_GROUP
	{
		.name = "uclamp.min",
		.flags = CFTYPE_NOT_ON_ROOT,
		.seq_show = cpu_uclamp_min_show,
		.write = cpu_uclamp_min_write,
	},
	{
		.name = "uclamp.max",
		.flags = CFTYPE_NOT_ON_ROOT,
		.seq_show = cpu_uclamp_max_show,
		.write = cpu_uclamp_max_write,
	},
#endif /* CONFIG_UCLAMP_TASK_GROUP */
	{ }	/* terminate */
};

struct cgroup_subsys cpu_cgrp_subsys = {
	.css_alloc	= cpu_cgroup_css_alloc,
	.css_online	= cpu_cgroup_css_online,
	.css_offline	= cpu_cgroup_css_offline,
	.css_released	= cpu_cgroup_css_released,
	.css_free	= cpu_cgroup_css_free,
	.css_extra_stat_show = cpu_extra_stat_show,
	.css_local_stat_show = cpu_local_stat_show,
	.can_attach	= cpu_cgroup_can_attach,
	.attach		= cpu_cgroup_attach,
	.cancel_attach	= cpu_cgroup_cancel_attach,
	.legacy_cftypes	= cpu_legacy_files,
	.dfl_cftypes	= cpu_files,
	.early_init	= true,
	.threaded	= true,
};

#endif /* CONFIG_CGROUP_SCHED */

/*
 * dump_cpu_task - 优先通过中断寄存器或远端回溯转储 CPU 状态，
 * 失败时输出其当前任务。
 */
void dump_cpu_task(int cpu)
{
	if (in_hardirq() && cpu == smp_processor_id()) {
		struct pt_regs *regs;

		regs = get_irq_regs();
		if (regs) {
			show_regs(regs);
			return;
		}
	}

	if (trigger_single_cpu_backtrace(cpu))
		return;

	pr_info("Task dump for CPU %d:\n", cpu);
	sched_show_task(cpu_curr(cpu));
}

/*
 * Nice levels are multiplicative, with a gentle 10% change for every
 * nice level changed. I.e. when a CPU-bound task goes from nice 0 to
 * nice 1, it will get ~10% less CPU time than another CPU-bound task
 * that remained on nice 0.
 *
 * The "10% effect" is relative and cumulative: from _any_ nice level,
 * if you go up 1 level, it's -10% CPU usage, if you go down 1 level
 * it's +10% CPU usage. (to achieve that we use a multiplier of 1.25.
 * If a task goes up by ~10% and another task goes down by ~10% then
 * the relative distance between them is ~25%.)
 */
const int sched_prio_to_weight[40] = {
 /* -20 */     88761,     71755,     56483,     46273,     36291,
 /* -15 */     29154,     23254,     18705,     14949,     11916,
 /* -10 */      9548,      7620,      6100,      4904,      3906,
 /*  -5 */      3121,      2501,      1991,      1586,      1277,
 /*   0 */      1024,       820,       655,       526,       423,
 /*   5 */       335,       272,       215,       172,       137,
 /*  10 */       110,        87,        70,        56,        45,
 /*  15 */        36,        29,        23,        18,        15,
};

/*
 * Inverse (2^32/x) values of the sched_prio_to_weight[] array, pre-calculated.
 *
 * In cases where the weight does not change often, we can use the
 * pre-calculated inverse to speed up arithmetics by turning divisions
 * into multiplications:
 */
const u32 sched_prio_to_wmult[40] = {
 /* -20 */     48388,     59856,     76040,     92818,    118348,
 /* -15 */    147320,    184698,    229616,    287308,    360437,
 /* -10 */    449829,    563644,    704093,    875809,   1099582,
 /*  -5 */   1376151,   1717300,   2157191,   2708050,   3363326,
 /*   0 */   4194304,   5237765,   6557202,   8165337,  10153587,
 /*   5 */  12820798,  15790321,  19976592,  24970740,  31350126,
 /*  10 */  39045157,  49367440,  61356676,  76695844,  95443717,
 /*  15 */ 119304647, 148102320, 186737708, 238609294, 286331153,
};

/*
 * call_trace_sched_update_nr_running - 封装可运行任务数变化的跟踪事件调用。
 */
void call_trace_sched_update_nr_running(struct rq *rq, int count)
{
        trace_sched_update_nr_running_tp(rq, count);
}

#ifdef CONFIG_SCHED_MM_CID

/*
 * MM CID（地址空间并发 ID）按 mm 分配紧凑编号，供用户态使用
 * 同一地址空间内的并发执行槽位；它不是 PID、物理 CPU 编号或 ASID。
 * mm->pcpu_cid 保存各 CPU 对该 mm 的编号缓存，任务的 mm_cid
 * 记录当前编号，last_mm_cid 为迁移优化保存线索。
 * 编号可缓存、转移、回收，因此用户态不能把它当任务的永久身份。
 *
 * 本节两项工作：迁移时尽量复用旧编号，使编号集合靠近 0；
 * 定期回收不再活跃的缓存，避免少量线程遍历许多 CPU 后占满编号。
 * 分配/本地切换细节见 sched.h 的 mm_cid_get()/switch_mm_cid()。
 */

/*
 * @cid_lock: Guarantee forward-progress of cid allocation.
 *
 * Concurrency ID allocation within a bitmap is mostly lock-free. The cid_lock
 * is only used when contention is detected by the lock-free allocation so
 * forward progress can be guaranteed.
 */
DEFINE_RAW_SPINLOCK(cid_lock);

/*
 * @use_cid_lock: Select cid allocation behavior: lock-free vs spinlock.
 *
 * When @use_cid_lock is 0, the cid allocation is lock-free. When contention is
 * detected, it is set to 1 to ensure that all newly coming allocations are
 * serialized by @cid_lock until the allocation which detected contention
 * completes and sets @use_cid_lock back to 0. This guarantees forward progress
 * of a cid allocation.
 */
int use_cid_lock;

/*
 * mm_cid remote-clear implements a lock-free algorithm to clear per-mm/cpu cid
 * concurrently with respect to the execution of the source runqueue context
 * switch.
 *
 * There is one basic properties we want to guarantee here:
 *
 * (1) Remote-clear should _never_ mark a per-cpu cid UNSET when it is actively
 * used by a task. That would lead to concurrent allocation of the cid and
 * userspace corruption.
 *
 * Provide this guarantee by introducing a Dekker memory ordering to guarantee
 * that a pair of loads observe at least one of a pair of stores, which can be
 * shown as:
 *
 *      X = Y = 0
 *
 *      w[X]=1          w[Y]=1
 *      MB              MB
 *      r[Y]=y          r[X]=x
 *
 * Which guarantees that x==0 && y==0 is impossible. But rather than using
 * values 0 and 1, this algorithm cares about specific state transitions of the
 * runqueue current task (as updated by the scheduler context switch), and the
 * per-mm/cpu cid value.
 *
 * Let's introduce task (Y) which has task->mm == mm and task (N) which has
 * task->mm != mm for the rest of the discussion. There are two scheduler state
 * transitions on context switch we care about:
 *
 * (TSA) Store to rq->curr with transition from (N) to (Y)
 *
 * (TSB) Store to rq->curr with transition from (Y) to (N)
 *
 * On the remote-clear side, there is one transition we care about:
 *
 * (TMA) cmpxchg to *pcpu_cid to set the LAZY flag
 *
 * There is also a transition to UNSET state which can be performed from all
 * sides (scheduler, remote-clear). It is always performed with a cmpxchg which
 * guarantees that only a single thread will succeed:
 *
 * (TMB) cmpxchg to *pcpu_cid to mark UNSET
 *
 * Just to be clear, what we do _not_ want to happen is a transition to UNSET
 * when a thread is actively using the cid (property (1)).
 *
 * Let's looks at the relevant combinations of TSA/TSB, and TMA transitions.
 *
 * Scenario A) (TSA)+(TMA) (from next task perspective)
 *
 * CPU0                                      CPU1
 *
 * Context switch CS-1                       Remote-clear
 *   - store to rq->curr: (N)->(Y) (TSA)     - cmpxchg to *pcpu_id to LAZY (TMA)
 *                                             (implied barrier after cmpxchg)
 *   - switch_mm_cid()
 *     - memory barrier (see switch_mm_cid()
 *       comment explaining how this barrier
 *       is combined with other scheduler
 *       barriers)
 *     - mm_cid_get (next)
 *       - READ_ONCE(*pcpu_cid)              - rcu_dereference(src_rq->curr)
 *
 * This Dekker ensures that either task (Y) is observed by the
 * rcu_dereference() or the LAZY flag is observed by READ_ONCE(), or both are
 * observed.
 *
 * If task (Y) store is observed by rcu_dereference(), it means that there is
 * still an active task on the cpu. Remote-clear will therefore not transition
 * to UNSET, which fulfills property (1).
 *
 * If task (Y) is not observed, but the lazy flag is observed by READ_ONCE(),
 * it will move its state to UNSET, which clears the percpu cid perhaps
 * uselessly (which is not an issue for correctness). Because task (Y) is not
 * observed, CPU1 can move ahead to set the state to UNSET. Because moving
 * state to UNSET is done with a cmpxchg expecting that the old state has the
 * LAZY flag set, only one thread will successfully UNSET.
 *
 * If both states (LAZY flag and task (Y)) are observed, the thread on CPU0
 * will observe the LAZY flag and transition to UNSET (perhaps uselessly), and
 * CPU1 will observe task (Y) and do nothing more, which is fine.
 *
 * What we are effectively preventing with this Dekker is a scenario where
 * neither LAZY flag nor store (Y) are observed, which would fail property (1)
 * because this would UNSET a cid which is actively used.
 */

/*
 * sched_mm_cid_migrate_from - 记录任务迁出的 CPU，
 * 供目标 CPU 处理地址空间并发 ID 转移。
 */
void sched_mm_cid_migrate_from(struct task_struct *t)
{
	t->migrate_from_cpu = task_cpu(t);
}

/*
 * __sched_mm_cid_migrate_from_fetch_cid - 查找源 CPU 上可转移的并发 ID，
 * 避免移动仍被使用的 ID。
 * 返回候选 ID，无需或无法转移时返回 -1。
 */
static
int __sched_mm_cid_migrate_from_fetch_cid(struct rq *src_rq,
					  struct task_struct *t,
					  struct mm_cid *src_pcpu_cid)
{
	struct mm_struct *mm = t->mm;
	struct task_struct *src_task;
	int src_cid, last_mm_cid;

	if (!mm)
		return -1;

	last_mm_cid = t->last_mm_cid;
	/*
	 * If the migrated task has no last cid, or if the current
	 * task on src rq uses the cid, it means the source cid does not need
	 * to be moved to the destination cpu.
	 */
	if (last_mm_cid == -1)
		return -1;
	src_cid = READ_ONCE(src_pcpu_cid->cid);
	if (!mm_cid_is_valid(src_cid) || last_mm_cid != src_cid)
		return -1;

	/*
	 * If we observe an active task using the mm on this rq, it means we
	 * are not the last task to be migrated from this cpu for this mm, so
	 * there is no need to move src_cid to the destination cpu.
	 */
	guard(rcu)();
	src_task = rcu_dereference(src_rq->curr);
	if (READ_ONCE(src_task->mm_cid_active) && src_task->mm == mm) {
		t->last_mm_cid = -1;
		return -1;
	}

	return src_cid;
}

/*
 * __sched_mm_cid_migrate_from_try_steal_cid - 安全取得源 CPU 的 ID。
 * 使用延迟释放标志和原子状态转换。
 * 确认没有活跃使用者后解除源端绑定，失败返回 -1。
 */
static
int __sched_mm_cid_migrate_from_try_steal_cid(struct rq *src_rq,
					      struct task_struct *t,
					      struct mm_cid *src_pcpu_cid,
					      int src_cid)
{
	struct task_struct *src_task;
	struct mm_struct *mm = t->mm;
	int lazy_cid;

	if (src_cid == -1)
		return -1;

	/*
	 * Attempt to clear the source cpu cid to move it to the destination
	 * cpu.
	 */
	lazy_cid = mm_cid_set_lazy_put(src_cid);
	if (!try_cmpxchg(&src_pcpu_cid->cid, &src_cid, lazy_cid))
		return -1;

	/*
	 * The implicit barrier after cmpxchg per-mm/cpu cid before loading
	 * rq->curr->mm matches the scheduler barrier in context_switch()
	 * between store to rq->curr and load of prev and next task's
	 * per-mm/cpu cid.
	 *
	 * The implicit barrier after cmpxchg per-mm/cpu cid before loading
	 * rq->curr->mm_cid_active matches the barrier in
	 * sched_mm_cid_exit(), sched_mm_cid_before_execve(), and
	 * sched_mm_cid_after_execve() between store to t->mm_cid_active and
	 * load of per-mm/cpu cid.
	 */

	/*
	 * If we observe an active task using the mm on this rq after setting
	 * the lazy-put flag, this task will be responsible for transitioning
	 * from lazy-put flag set to MM_CID_UNSET.
	 */
	scoped_guard (rcu) {
		src_task = rcu_dereference(src_rq->curr);
		if (READ_ONCE(src_task->mm_cid_active) && src_task->mm == mm) {
			/*
			 * We observed an active task for this mm, there is therefore
			 * no point in moving this cid to the destination cpu.
			 */
			t->last_mm_cid = -1;
			return -1;
		}
	}

	/*
	 * The src_cid is unused, so it can be unset.
	 */
	if (!try_cmpxchg(&src_pcpu_cid->cid, &lazy_cid, MM_CID_UNSET))
		return -1;
	/*
	 * 此处只解除源 CPU 绑定，不归还分配位图：成功返回后由调用者
	 * 决定转给目标 CPU 还是释放；失败则不取得编号所有权。
	 */
	WRITE_ONCE(src_pcpu_cid->recent_cid, MM_CID_UNSET);
	return src_cid;
}

/*
 * Migration to dst cpu. Called with dst_rq lock held.
 * Interrupts are disabled, which keeps the window of cid ownership without the
 * source rq lock held small.
 */
/*
 * sched_mm_cid_migrate_to - 任务入目标队列时转移或回收源并发 ID，
 * 使分配保持紧凑。
 * 调用者须持有目标队列锁并关闭中断。
 */
void sched_mm_cid_migrate_to(struct rq *dst_rq, struct task_struct *t)
{
	struct mm_cid *src_pcpu_cid, *dst_pcpu_cid;
	struct mm_struct *mm = t->mm;
	int src_cid, src_cpu;
	bool dst_cid_is_set;
	struct rq *src_rq;

	lockdep_assert_rq_held(dst_rq);

	if (!mm)
		return;
	/*
	 * 编号缓存属于 (mm, CPU)，而非任务私有。例如同 mm 的另一线程
	 * 还在源 CPU 执行时，即使 t 已迁走，也不能把源编号拿给目标。
	 * fetch 只找候选，try_steal 才通过原子协议取得所有权；中间的
	 * 检查失败是允许的，后续上下文切换仍可走常规编号分配。
	 */
	src_cpu = t->migrate_from_cpu;
	if (src_cpu == -1) {
		t->last_mm_cid = -1;
		return;
	}
	/*
	 * Move the src cid if the dst cid is unset. This keeps id
	 * allocation closest to 0 in cases where few threads migrate around
	 * many CPUs.
	 *
	 * If destination cid or recent cid is already set, we may have
	 * to just clear the src cid to ensure compactness in frequent
	 * migrations scenarios.
	 *
	 * It is not useful to clear the src cid when the number of threads is
	 * greater or equal to the number of allowed CPUs, because user-space
	 * can expect that the number of allowed cids can reach the number of
	 * allowed CPUs.
	 */
	dst_pcpu_cid = per_cpu_ptr(mm->pcpu_cid, cpu_of(dst_rq));
	dst_cid_is_set = !mm_cid_is_unset(READ_ONCE(dst_pcpu_cid->cid)) ||
			 !mm_cid_is_unset(READ_ONCE(dst_pcpu_cid->recent_cid));
	if (dst_cid_is_set && atomic_read(&mm->mm_users) >= READ_ONCE(mm->nr_cpus_allowed))
		return;
	src_pcpu_cid = per_cpu_ptr(mm->pcpu_cid, src_cpu);
	src_rq = cpu_rq(src_cpu);
	src_cid = __sched_mm_cid_migrate_from_fetch_cid(src_rq, t, src_pcpu_cid);
	if (src_cid == -1)
		return;
	src_cid = __sched_mm_cid_migrate_from_try_steal_cid(src_rq, t, src_pcpu_cid,
							    src_cid);
	if (src_cid == -1)
		return;
	if (dst_cid_is_set) {
		/* 目标已有编号，只回收源编号，不覆盖目标缓存。 */
		__mm_cid_put(mm, src_cid);
		return;
	}
	/* Move src_cid to dst cpu. */
	mm_cid_snapshot_time(dst_rq, mm);
	WRITE_ONCE(dst_pcpu_cid->cid, src_cid);
	WRITE_ONCE(dst_pcpu_cid->recent_cid, src_cid);
}

/*
 * sched_mm_cid_remote_clear - 远程回收地址空间在指定 CPU 上
 * 未被活跃任务使用的并发 ID。
 * 通过延迟释放标志和内存序配合上下文切换，避免回收正在使用的 ID。
 */
static void sched_mm_cid_remote_clear(struct mm_struct *mm, struct mm_cid *pcpu_cid,
				      int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	struct task_struct *t;
	int cid, lazy_cid;

	cid = READ_ONCE(pcpu_cid->cid);
	if (!mm_cid_is_valid(cid))
		return;

	/*
	 * Clear the cpu cid if it is set to keep cid allocation compact.  If
	 * there happens to be other tasks left on the source cpu using this
	 * mm, the next task using this mm will reallocate its cid on context
	 * switch.
	 *
	 * 回收分两阶段：有效 CID -> 带 LAZY 标志 -> MM_CID_UNSET。
	 * 先标 LAZY 再观察 rq->curr，与切换方“先发布 curr 再读 CID”
	 * 的屏障配对，保证至少一方看见另一方的更新：要么回收方看见
	 * 活跃使用者而放弃，要么切换方看见 LAZY 并接手解除绑定。
	 * 因此不能直接把首次读到的 CID 清空并归还给分配位图。
	 */
	lazy_cid = mm_cid_set_lazy_put(cid);
	if (!try_cmpxchg(&pcpu_cid->cid, &cid, lazy_cid))
		return;

	/*
	 * The implicit barrier after cmpxchg per-mm/cpu cid before loading
	 * rq->curr->mm matches the scheduler barrier in context_switch()
	 * between store to rq->curr and load of prev and next task's
	 * per-mm/cpu cid.
	 *
	 * The implicit barrier after cmpxchg per-mm/cpu cid before loading
	 * rq->curr->mm_cid_active matches the barrier in
	 * sched_mm_cid_exit(), sched_mm_cid_before_execve(), and
	 * sched_mm_cid_after_execve() between store to t->mm_cid_active and
	 * load of per-mm/cpu cid.
	 */

	/*
	 * If we observe an active task using the mm on this rq after setting
	 * the lazy-put flag, that task will be responsible for transitioning
	 * from lazy-put flag set to MM_CID_UNSET.
	 */
	scoped_guard (rcu) {
		t = rcu_dereference(rq->curr);
		if (READ_ONCE(t->mm_cid_active) && t->mm == mm)
			return;
	}

	/*
	 * The cid is unused, so it can be unset.
	 * Disable interrupts to keep the window of cid ownership without rq
	 * lock small.
	 *
	 * 本地切换与远端回收都可能完成 LAZY -> UNSET；只有 CAS
	 * 成功的一方才能归还位图中的 CID，避免同一编号被重复释放。
	 */
	scoped_guard (irqsave) {
		if (try_cmpxchg(&pcpu_cid->cid, &lazy_cid, MM_CID_UNSET))
			__mm_cid_put(mm, cid);
	}
}

/*
 * sched_mm_cid_remote_clear_old - 刷新活跃并发 ID 的时间戳，
 * 或尝试回收长时间未使用的 ID。
 */
static void sched_mm_cid_remote_clear_old(struct mm_struct *mm, int cpu)
{
	struct rq *rq = cpu_rq(cpu);
	struct mm_cid *pcpu_cid;
	struct task_struct *curr;
	u64 rq_clock;

	/*
	 * rq->clock load is racy on 32-bit but one spurious clear once in a
	 * while is irrelevant.
	 */
	rq_clock = READ_ONCE(rq->clock);
	pcpu_cid = per_cpu_ptr(mm->pcpu_cid, cpu);

	/*
	 * In order to take care of infrequently scheduled tasks, bump the time
	 * snapshot associated with this cid if an active task using the mm is
	 * observed on this rq.
	 */
	scoped_guard (rcu) {
		curr = rcu_dereference(rq->curr);
		if (READ_ONCE(curr->mm_cid_active) && curr->mm == mm) {
			WRITE_ONCE(pcpu_cid->time, rq_clock);
			return;
		}
	}

	if (rq_clock < pcpu_cid->time + SCHED_MM_CID_PERIOD_NS)
		return;
	sched_mm_cid_remote_clear(mm, pcpu_cid, cpu);
}

/*
 * sched_mm_cid_remote_clear_weight - 尝试回收不小于当前 ID 集合大小的高编号 ID，
 * 促进编号紧凑。
 */
static void sched_mm_cid_remote_clear_weight(struct mm_struct *mm, int cpu,
					     int weight)
{
	struct mm_cid *pcpu_cid;
	int cid;

	pcpu_cid = per_cpu_ptr(mm->pcpu_cid, cpu);
	cid = READ_ONCE(pcpu_cid->cid);
	if (!mm_cid_is_valid(cid) || cid < weight)
		return;
	sched_mm_cid_remote_clear(mm, pcpu_cid, cpu);
}

/*
 * task_mm_cid_work - 在任务工作上下文限频扫描并发 ID，
 * 回收过期和高编号的闲置 ID。
 */
static void task_mm_cid_work(struct callback_head *work)
{
	unsigned long now = jiffies, old_scan, next_scan;
	struct task_struct *t = current;
	struct cpumask *cidmask;
	struct mm_struct *mm;
	int weight, cpu;

	WARN_ON_ONCE(t != container_of(work, struct task_struct, cid_work));

	work->next = work;	/* Prevent double-add */
	if (t->flags & PF_EXITING)
		return;
	mm = t->mm;
	if (!mm)
		return;
	old_scan = READ_ONCE(mm->mm_cid_next_scan);
	next_scan = now + msecs_to_jiffies(MM_CID_SCAN_DELAY);
	if (!old_scan) {
		unsigned long res;

		res = cmpxchg(&mm->mm_cid_next_scan, old_scan, next_scan);
		if (res != old_scan)
			old_scan = res;
		else
			old_scan = next_scan;
	}
	if (time_before(now, old_scan))
		return;
	/*
	 * 同一 mm 的多个线程可能同时到期；用共享截止时间的 CAS 认领
	 * 本轮扫描，只有胜者遍历所有 CPU，避免线程数放大扫描开销。
	 */
	if (!try_cmpxchg(&mm->mm_cid_next_scan, &old_scan, next_scan))
		return;
	cidmask = mm_cidmask(mm);
	/* Clear cids that were not recently used. */
	for_each_possible_cpu(cpu)
		sched_mm_cid_remote_clear_old(mm, cpu);
	weight = cpumask_weight(cidmask);
	/*
	 * 若已分配 N 个编号却存在编号 >= N，说明集合有空洞。
	 * 尝试撤掉高位缓存，让后续分配优先填低位；仍活跃的编号不会
	 * 被强行释放，所以紧凑化是逐步完成，不要求每轮扫描后连续。
	 */
	/*
	 * Clear cids that are greater or equal to the cidmask weight to
	 * recompact it.
	 */
	for_each_possible_cpu(cpu)
		sched_mm_cid_remote_clear_weight(mm, cpu, weight);
}

/*
 * init_sched_mm_cid - 初始化任务的并发 ID 扫描工作项，
 * 并为新地址空间安排首次扫描时间。
 */
void init_sched_mm_cid(struct task_struct *t)
{
	struct mm_struct *mm = t->mm;
	int mm_users = 0;

	if (mm) {
		mm_users = atomic_read(&mm->mm_users);
		if (mm_users == 1)
			mm->mm_cid_next_scan = jiffies + msecs_to_jiffies(MM_CID_SCAN_DELAY);
	}
	t->cid_work.next = &t->cid_work;	/* Protect against double add */
	init_task_work(&t->cid_work, task_mm_cid_work);
}

/*
 * task_tick_mm_cid - 调度 tick 中检查扫描期限，
 * 为合适的用户任务提交并发 ID 回收工作。
 */
void task_tick_mm_cid(struct rq *rq, struct task_struct *curr)
{
	struct callback_head *work = &curr->cid_work;
	unsigned long now = jiffies;

	if (!curr->mm || (curr->flags & (PF_EXITING | PF_KTHREAD)) ||
	    work->next != work)
		return;
	if (time_before(now, READ_ONCE(curr->mm->mm_cid_next_scan)))
		return;

	/*
	 * tick 只提交工作，扫描所有 CPU 的操作放到该任务返回用户态前
	 * 执行。work->next 自指表示尚未排队，防止每个 tick 重复添加。
	 */
	/* No page allocation under rq lock */
	task_work_add(curr, work, TWA_RESUME);
}

/*
 * sched_mm_cid_exit - 任务退出时停止使用地址空间并发 ID，
 * 按所需内存序释放并清空任务记录。
 */
void sched_mm_cid_exit(struct task_struct *t)
{
	struct mm_struct *mm = t->mm;
	struct rq *rq;

	if (!mm)
		return;

	preempt_disable();
	rq = this_rq();
	guard(rq_lock_irqsave)(rq);
	preempt_enable_no_resched();	/* holding spinlock */
	WRITE_ONCE(t->mm_cid_active, 0);
	/*
	 * Store t->mm_cid_active before loading per-mm/cpu cid.
	 * Matches barrier in sched_mm_cid_remote_clear_old().
	 */
	smp_mb();
	mm_cid_put(mm);
	t->last_mm_cid = t->mm_cid = -1;
}

/*
 * sched_mm_cid_before_execve - exec 更换地址空间之前停用并释放旧并发 ID。
 */
void sched_mm_cid_before_execve(struct task_struct *t)
{
	struct mm_struct *mm = t->mm;
	struct rq *rq;

	if (!mm)
		return;

	preempt_disable();
	rq = this_rq();
	guard(rq_lock_irqsave)(rq);
	preempt_enable_no_resched();	/* holding spinlock */
	WRITE_ONCE(t->mm_cid_active, 0);
	/*
	 * Store t->mm_cid_active before loading per-mm/cpu cid.
	 * Matches barrier in sched_mm_cid_remote_clear_old().
	 */
	smp_mb();
	mm_cid_put(mm);
	t->last_mm_cid = t->mm_cid = -1;
}

/*
 * sched_mm_cid_after_execve - exec 地址空间切换完成后恢复并发 ID 活动状态，
 * 并取得新 ID。
 */
void sched_mm_cid_after_execve(struct task_struct *t)
{
	struct mm_struct *mm = t->mm;
	struct rq *rq;

	if (!mm)
		return;

	preempt_disable();
	rq = this_rq();
	scoped_guard (rq_lock_irqsave, rq) {
		preempt_enable_no_resched();	/* holding spinlock */
		WRITE_ONCE(t->mm_cid_active, 1);
		/*
		 * Store t->mm_cid_active before loading per-mm/cpu cid.
		 * Matches barrier in sched_mm_cid_remote_clear_old().
		 */
		smp_mb();
		t->last_mm_cid = t->mm_cid = mm_cid_get(rq, t, mm);
	}
}

/*
 * sched_mm_cid_fork - 校验新用户任务尚未分配并发 ID，
 * 并启用其并发 ID 调度管理。
 */
void sched_mm_cid_fork(struct task_struct *t)
{
	WARN_ON_ONCE(!t->mm || t->mm_cid != -1);
	t->mm_cid_active = 1;
}
#endif /* CONFIG_SCHED_MM_CID */

static DEFINE_PER_CPU(struct sched_change_ctx, sched_change_ctx);

/*
 * sched_change_begin - 保存任务的入队及运行状态，
 * 并临时出队，为修改调度属性建立上下文。
 * 调用者须持有队列锁；返回每 CPU 上下文，
 * 须与 sched_change_end 配对且不可嵌套。
 */
struct sched_change_ctx *sched_change_begin(struct task_struct *p, unsigned int flags)
{
	struct sched_change_ctx *ctx = this_cpu_ptr(&sched_change_ctx);
	struct rq *rq = task_rq(p);

	lockdep_assert_rq_held(rq);

	/*
	 * 每 CPU 仅一份 ctx，调用者必须在同一非迁移持锁区间成对使用。
	 * queued 与 running 分开保存：当前 donor 与“在队列上”不是
	 * 同一问题，修改属性后分别恢复队列位置和类的当前实体状态。
	 * scoped_guard(sched_change, ...) 利用 sched.h 中的 DEFINE_CLASS
	 * 在离开作用域时自动调用 end；阅读这类调用时，闭合花括号也就
	 * 是重新入队/恢复当前实体的时刻，不能只找显式的 end 调用。
	 */
	*ctx = (struct sched_change_ctx){
		.p = p,
		.flags = flags,
		.queued = task_on_rq_queued(p),
		.running = task_current_donor(rq, p),
	};

	if (ctx->queued)
		dequeue_task(rq, p, flags);
	if (ctx->running)
		put_prev_task(rq, p);

	return ctx;
}

/*
 * sched_change_end - 调度属性修改完成后，
 * 依据保存的上下文恢复任务入队和当前运行状态。
 * 调用者须继续持有任务运行队列锁。
 */
void sched_change_end(struct sched_change_ctx *ctx)
{
	struct task_struct *p = ctx->p;
	struct rq *rq = task_rq(p);

	lockdep_assert_rq_held(rq);

	/*
	 * begin 传入的 SAVE/MOVE 等位与 ENQUEUE 对应位配合使用；
	 * 调用者可在作用域内追加 HEAD/REPLENISH 等入队要求。按新的
	 * sched_class 恢复，故属性/类修改必须发生在 begin 和 end 之间。
	 */
	if (ctx->queued)
		enqueue_task(rq, p, ctx->flags | ENQUEUE_NOCLOCK);
	if (ctx->running)
		set_next_task(rq, p);
}
