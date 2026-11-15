//! One drafter body pass for two or three conversations (CANADA_SCRATCH_BATCHED_DRAFT_BODY=1). Each stage's
//! residual boundaries, routed and shared experts and both sums run once over all the conversations' five-row
//! blocks (the weights are read once a pass); each conversation's attention runs on its own window, on its
//! five rows of the shared boundary input. Heads, Markov correction and proposal sampling stay per conversation
//! (two heads pool as in dspark_pair_query). The pass follows pico-dgx-ds4's one-pass drafter for batch steps
//! (docs/batching.md, B3 phase 2a). Drafts only propose; the verifier decides every token.
use super::*;
use anyhow::Context;

impl QueryNetwork {
    /// Whether this drafter's head pass can share one NVFP4 pass with `other`'s (config::draft_head_pool): both
    /// run the rank's one NVFP4 copy, in the same form, and neither gathers its base logits.
    fn can_pool_nvfp4_head(&self, other: &Self) -> bool {
        self.private_head_active && other.private_head_active && self.head_subset == other.head_subset
            && !self.gather_base_logits && !other.gather_base_logits
            && self.private_head.as_ref().zip(other.private_head.as_ref())
                .is_some_and(|(a, b)| a.nvfp4_pool_available() && a.shares_nvfp4(b))
    }
}
struct BatchStage {
    attn_boundary: Boundary,
    ffn_boundary: Boundary,
    ffn: LayerFfn,
    partial: Buffer<bf16>,
    attn_reduced: Buffer<bf16>,
    ffn_reduced: Buffer<bf16>,
}
/// The shared part of a pass over `rows` = 5 x conversations rows.
pub(crate) struct BatchBody {
    rows: usize,
    stages: Vec<BatchStage>,
    workspace: Workspace,
    tokens: TokenRows,
    input: Buffer<bf16>,
    combined: Buffer<bf16>,
    normalized: Buffer<bf16>,
    /// One captured pass per ordered set of conversations (their normalized outputs name them).
    graphs: Vec<(Vec<u64>, Graph)>,
    /// config::batched_draft_interleave: a capture's check found the phase order changing the output (kept off
    /// from then on), and whether the first passing check was reported.
    interleave_refused: bool,
    interleave_reported: bool,
}
/// The passes of two and of three conversations, shared by every drafter forked from one origin.
pub(crate) struct BatchBodies {
    bodies: Vec<BatchBody>,
}
impl BatchBodies {
    pub(super) fn new(origin: &QueryNetwork, stream: &Stream, kernels: &Path) -> Result<Self> {
        Ok(Self { bodies: vec![BatchBody::new(origin, 10, stream, kernels)?, BatchBody::new(origin, 15, stream, kernels)?] })
    }
    /// One pass over `nets` (two or three independent drafters of one origin), each from its own anchor.
    pub(crate) fn run(&mut self, nets: &mut [&mut QueryNetwork], anchors: &[u32], comm: &Roce, stream: &Stream)
        -> Result<()> {
        let body = self.bodies.iter_mut().find(|b| b.rows == QUERY_ROWS * nets.len())
            .context("a batched drafter pass takes two or three conversations")?;
        body.run(nets, anchors, comm, stream)
    }
}
impl BatchBody {
    fn new(origin: &QueryNetwork, rows: usize, stream: &Stream, kernels: &Path) -> Result<Self> {
        origin.guard(stream)?;
        ensure!(crate::config::batched_draft_body() && rows % QUERY_ROWS == 0 && origin.stages.len() == 3
            && origin.prepared_layouts.len() == 3, "batched drafter body owner");
        let cuda = &origin.cuda;
        stream.join_default()?;
        let mut stages = Vec::with_capacity(3);
        for (stage, layout) in origin.prepared_layouts.iter().cloned().enumerate() {
            let id = LayerId::draft(stage)?;
            stages.push(BatchStage {
                attn_boundary: Boundary::new_named(cuda, stream, origin.weights.clone(), id, "attn", rows, kernels)?,
                ffn_boundary: Boundary::new_named(cuda, stream, origin.weights.clone(), id, "ffn", rows, kernels)?,
                ffn: LayerFfn::new_draft_batch(cuda, origin.weights.clone(), layout, kernels, rows,
                    &origin.stages[stage].ffn)?,
                partial: cuda.allocate(rows * 5120)?,
                attn_reduced: cuda.allocate(rows * 5120)?,
                ffn_reduced: cuda.allocate(rows * 5120)?,
            });
        }
        let body = Self {
            rows,
            stages,
            workspace: Workspace::new_draft_rows(cuda, &kernels.join("moe-runtime-dspark-m16-v1"), rows)?,
            tokens: TokenRows::new(cuda, &vec![128799; rows], rows)?,
            input: cuda.allocate(rows * 20480)?,
            combined: cuda.allocate(rows * 5120)?,
            normalized: cuda.allocate(rows * 5120)?,
            graphs: Vec::new(),
            interleave_refused: false,
            interleave_reported: false,
        };
        stream.synchronize()?;
        eprintln!("rank{} batched drafter body M{rows} prepared", origin.rank);
        Ok(body)
    }
    /// `interleave` (config::batched_draft_interleave): the conversations' attention sublayers phase by phase.
    fn enqueue(&mut self, nets: &mut [&mut QueryNetwork], comm: &Roce, stream: &Stream, interleave: bool) -> Result<()> {
        let cuda = nets[0].cuda.clone();
        let block = QUERY_ROWS * 5120;
        for (k, net) in nets.iter_mut().enumerate() {
            let net = &mut **net;
            net.io.run_input(&net.tokens, comm, stream)?;
            ensure!(net.io.input().len() == QUERY_ROWS * 20480, "drafter input block extent");
            unsafe { cuda.copy_device(self.input.ptr() + (k * QUERY_ROWS * 20480 * 2) as u64,
                net.io.input().ptr(), QUERY_ROWS * 20480 * 2, stream) }?;
        }
        for stage in 0..3 {
            let (previous, current) = self.stages.split_at_mut(stage);
            let s = &mut current[0];
            let (residual, pre) = match previous.last() {
                Some(p) => (p.ffn_boundary.output(), Some(p.ffn_boundary.pre())),
                None => (&self.input, None),
            };
            s.attn_boundary.prepare(residual, pre, stream)?;
            if interleave {
                // Every conversation's input projection, then every KV projection, query projection, attention
                // with wo_a, wo_b: each phase's weights are read from memory by the first conversation only.
                for phase in [1u8, 2, 4, 8, 16] {
                    for (k, net) in nets.iter_mut().enumerate() {
                        let net = &mut **net;
                        // Safety: a view of this stage's boundary input, alive for the pass.
                        let rows = unsafe { s.attn_boundary.input().view(k * block, block) }?;
                        let (cache, candidates, locations) = net.state.query_parts(stage, stream)?;
                        net.stages[stage].run_attention_phase(phase, &rows, &net.positions, locations, cache, candidates, stream)?;
                    }
                }
                for (k, net) in nets.iter().enumerate() {
                    let partial = net.stages[stage].attention_partial();
                    ensure!(partial.len() == block, "drafter attention partial extent");
                    unsafe { cuda.copy_device(s.partial.ptr() + (k * block * 2) as u64, partial.ptr(), block * 2, stream) }?;
                }
            } else {
            for (k, net) in nets.iter_mut().enumerate() {
                let net = &mut **net;
                // Safety: a view of this stage's boundary input, alive for the pass.
                let rows = unsafe { s.attn_boundary.input().view(k * block, block) }?;
                let (cache, candidates, locations) = net.state.query_parts(stage, stream)?;
                let partial = net.stages[stage].run_attention(&rows, &net.positions, locations, cache, candidates,
                    comm, stream)?;
                ensure!(partial.len() == block, "drafter attention partial extent");
                unsafe { cuda.copy_device(s.partial.ptr() + (k * block * 2) as u64, partial.ptr(), block * 2, stream) }?;
            }
            }
            unsafe { comm.sum_bf16(&s.partial, &mut s.attn_reduced, stream) }?;
            s.attn_boundary.finish(&s.attn_reduced, residual, stream)?;
            s.ffn_boundary.prepare(s.attn_boundary.output(), Some(s.attn_boundary.pre()), stream)?;
            s.ffn.run(&self.workspace, s.ffn_boundary.input(), &self.tokens, stream)?;
            unsafe { comm.sum_bf16(s.ffn.output(), &mut s.ffn_reduced, stream) }?;
            s.ffn_boundary.finish(&s.ffn_reduced, s.attn_boundary.output(), stream)?;
        }
        let last = self.stages.last().unwrap();
        let (mut input, mut pre, mut output) = (last.ffn_boundary.output().ptr(), last.ffn_boundary.pre().ptr(),
            self.combined.ptr());
        let mut args = [(&mut input as *mut u64).cast(), (&mut pre as *mut u64).cast(), (&mut output as *mut u64).cast()];
        unsafe { nets[0].module.function("ca_final_hc_combine")?.launch([self.rows as u32, 1, 1], [256, 1, 1], 0,
            stream, &mut args) }?;
        nets[0].norm.rms(&self.combined, &nets[0].norm_weight, &mut self.normalized, self.rows, 5120, stream)?;
        for (k, net) in nets.iter_mut().enumerate() {
            unsafe { cuda.copy_device(net.normalized.ptr(), self.normalized.ptr() + (k * block * 2) as u64,
                block * 2, stream) }?;
            // The Markov and confidence heads read a conversation's own post-mix rows (QueryNetwork::combined,
            // bound by pointer in their graphs). Until s1503 only the normalized rows were handed back, so after a
            // batched body those heads saw the rows of the conversation's last pass of its own: stale confidences
            // (proposals stay exact; the rows a step verifies no longer followed the current draft).
            unsafe { cuda.copy_device(net.combined.ptr(), self.combined.ptr() + (k * block * 2) as u64,
                block * 2, stream) }?;
        }
        Ok(())
    }
    fn run(&mut self, nets: &mut [&mut QueryNetwork], anchors: &[u32], comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(nets.len() * QUERY_ROWS == self.rows && anchors.len() == nets.len()
            && anchors.iter().all(|a| *a < 129280), "batched drafter pass extent");
        for i in 0..nets.len() {
            nets[i].guard(stream)?;
            ensure!(nets[i].rank == comm.rank() && nets[i].comm_instance.is_none_or(|c| c == comm.instance())
                && Rc::ptr_eq(&nets[i].weights, &nets[0].weights), "batched drafters of one origin and communicator");
            for j in 0..i {
                ensure!(nets[i].state.independent_from(&nets[j].state)?
                    && nets[i].normalized.ptr() != nets[j].normalized.ptr()
                    && nets[i].io.local_logits().ptr() != nets[j].io.local_logits().ptr(),
                    "batched drafters need independent owners");
            }
        }
        let result = (|| -> Result<()> {
            let mut ids = Vec::with_capacity(self.rows);
            for (net, &anchor) in nets.iter_mut().zip(anchors) {
                net.ready = false;
                net.last_anchor = None;
                net.markov.invalidate_results();
                let layout = net.state.prepare_query(stream)?;
                net.positions.update(&layout.positions)?;
                let block = [i64::from(anchor), 128799, 128799, 128799, 128799];
                net.tokens.update(&block, QUERY_ROWS)?;
                ids.extend(block);
            }
            self.tokens.update(&ids, self.rows)?;
            comm.select_decode_reduction(true, stream)?;
            comm.select_prefill_rows(QUERY_ROWS, QUERY_ROWS, stream)?;
            stream.join_default()?;
            let key: Vec<u64> = nets.iter().map(|n| n.normalized.ptr()).collect();
            let index = match self.graphs.iter().position(|(k, _)| *k == key) {
                Some(index) => index,
                None => {
                    let mut interleave = crate::config::batched_draft_interleave() && !self.interleave_refused;
                    self.enqueue(nets, comm, stream, false)?;
                    stream.synchronize()?;
                    comm.health()?;
                    if interleave {
                        // On this set's own inputs: the pass in conversation order (above), then phase by phase.
                        // Every conversation's normalized rows must agree bit for bit (a drafter's attention
                        // operators would differ if they shared scratch with another conversation's).
                        let reference = self.normalized.download()?;
                        self.enqueue(nets, comm, stream, true)?;
                        stream.synchronize()?;
                        comm.health()?;
                        let candidate = self.normalized.download()?;
                        if reference.iter().zip(&candidate).all(|(a, b)| a.to_bits() == b.to_bits()) {
                            if !self.interleave_reported {
                                eprintln!("rank{} batched drafter body M{}: phase order checked against conversation order, identical output",
                                    comm.rank(), self.rows);
                                self.interleave_reported = true;
                            }
                        } else {
                            eprintln!("rank{} batched drafter body M{}: the phase order changed the pass's output; keeping conversation order",
                                comm.rank(), self.rows);
                            self.interleave_refused = true;
                            interleave = false;
                            self.enqueue(nets, comm, stream, false)?;
                            stream.synchronize()?;
                            comm.health()?;
                        }
                    }
                    let graph = stream.capture(|| self.enqueue(nets, comm, stream, interleave))?;
                    if self.graphs.len() == 8 { self.graphs.remove(0); }
                    self.graphs.push((key, graph));
                    for net in nets.iter_mut() { net.comm_instance = Some(comm.instance()); }
                    self.graphs.len() - 1
                }
            };
            unsafe { self.graphs[index].1.launch(stream) }?;
            // Heads: adjacent owners pool where their heads allow it; the rest run their own.
            let mut k = 0;
            while k < nets.len() {
                // config::draft_head_pool: up to three adjacent conversations that run the rank's NVFP4 copy in the
                // same form share one pass over it.
                let mut pooled = 1;
                while pooled < 3 && k + pooled < nets.len() && nets[k].can_pool_nvfp4_head(&*nets[k + pooled]) { pooled += 1; }
                if pooled >= 2 {
                    let head = nets[k].private_head.as_ref().ok_or_else(|| anyhow::anyhow!("private draft head absent"))?;
                    head.set_nvfp4_subset_mode(nets[k].head_subset)?;
                    let mut parts = Vec::with_capacity(pooled);
                    for net in nets[k..k + pooled].iter() {
                        let own = net.private_head.as_ref().ok_or_else(|| anyhow::anyhow!("private draft head absent"))?;
                        parts.push((&net.normalized, net.io.private_draft_head_output(own, stream)?));
                    }
                    head.run_nvfp4_pooled(&parts, stream)?;
                    BATCHED_HEAD_FORMS[usize::from(nets[k].head_subset)].fetch_add(pooled as u64, std::sync::atomic::Ordering::Relaxed);
                    BATCHED_HEAD_POOLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    k += pooled;
                    continue;
                }
                if k + 1 < nets.len() && nets[k].can_pool_query_head(&*nets[k + 1]) {
                    let (a, b) = nets.split_at_mut(k + 1);
                    let (a, b) = (&mut *a[k], &mut *b[0]);
                    a.pair_head.as_mut().unwrap().run(
                        [a.private_head.as_ref().unwrap(), b.private_head.as_ref().unwrap()],
                        [&a.normalized, &b.normalized], [a.io.local_logits(), b.io.local_logits()], stream)?;
                    k += 2;
                } else {
                    nets[k].run_head(comm, stream)?;
                    BATCHED_HEAD_FORMS[usize::from(nets[k].head_subset)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    k += 1;
                }
            }
            for net in nets.iter_mut() { net.state.publish_queries(stream)?; }
            comm.health()?;
            Ok(())
        })();
        for (net, &anchor) in nets.iter_mut().zip(anchors) {
            if result.is_err() { net.poisoned = true; net.state.invalidate(); }
            else { net.ready = true; net.last_anchor = Some(anchor); }
        }
        result
    }
}
fn bits(buffer: &Buffer<bf16>) -> Result<Vec<u16>> {
    Ok(buffer.download()?.iter().map(|x| x.to_bits()).collect())
}
impl BatchBody {
    /// Screen: owner `k`'s five rows of every stage buffer of the last pass, in pass order (the attention
    /// partial is the owner's own).
    fn stage_snapshot(&self, k: usize, net: &QueryNetwork) -> Result<Vec<(String, Vec<u16>)>> {
        let slice = |buffer: &Buffer<bf16>| -> Result<Vec<u16>> {
            let all = bits(buffer)?;
            ensure!(all.len() % self.rows == 0, "batched drafter stage buffer extent");
            let width = all.len() / self.rows;
            Ok(all[k * QUERY_ROWS * width..(k + 1) * QUERY_ROWS * width].to_vec())
        };
        let mut out = vec![("input".to_owned(), slice(&self.input)?)];
        for (stage, s) in self.stages.iter().enumerate() {
            out.push((format!("stage {stage} attention input"), slice(s.attn_boundary.input())?));
            out.push((format!("stage {stage} attention partial"), bits(net.stages[stage].attention.output())?));
            out.push((format!("stage {stage} attention sum"), slice(&s.attn_reduced)?));
            out.push((format!("stage {stage} attention boundary output"), slice(s.attn_boundary.output())?));
            out.push((format!("stage {stage} expert input"), slice(s.ffn_boundary.input())?));
            out.push((format!("stage {stage} expert partial"), slice(s.ffn.output())?));
            out.push((format!("stage {stage} expert sum"), slice(&s.ffn_reduced)?));
            out.push((format!("stage {stage} expert boundary output"), slice(s.ffn_boundary.output())?));
        }
        out.push(("combined".to_owned(), slice(&self.combined)?));
        out.push(("normalized".to_owned(), slice(&self.normalized)?));
        Ok(out)
    }
}
impl QueryNetwork {
    /// Screen: every stage buffer of this owner's last separate pass, in pass order.
    fn stage_snapshot(&self) -> Result<Vec<(String, Vec<u16>)>> {
        let mut out = vec![("input".to_owned(), bits(self.io.input())?)];
        for (stage, s) in self.stages.iter().enumerate() {
            out.push((format!("stage {stage} attention input"), bits(s.attn_boundary.input())?));
            out.push((format!("stage {stage} attention partial"), bits(s.attention.output())?));
            out.push((format!("stage {stage} attention sum"), bits(&s.attn_reduced)?));
            out.push((format!("stage {stage} attention boundary output"), bits(s.attn_boundary.output())?));
            out.push((format!("stage {stage} expert input"), bits(s.ffn_boundary.input())?));
            out.push((format!("stage {stage} expert partial"), bits(s.ffn.output())?));
            out.push((format!("stage {stage} expert sum"), bits(&s.ffn_reduced)?));
            out.push((format!("stage {stage} expert boundary output"), bits(s.ffn_boundary.output())?));
        }
        out.push(("combined".to_owned(), bits(&self.combined)?));
        out.push(("normalized".to_owned(), bits(&self.normalized)?));
        Ok(out)
    }
    /// The body passes this drafter shares with its origin's other forks, when prepared.
    pub(crate) fn batch_bodies(&self) -> Option<Rc<std::cell::OnceCell<std::cell::RefCell<BatchBodies>>>> {
        self.shared_bodies.get().is_some().then(|| self.shared_bodies.clone())
    }
    pub(crate) fn shares_batch_bodies(&self, other: &Self) -> bool {
        self.shared_bodies.get().is_some() && Rc::ptr_eq(&self.shared_bodies, &other.shared_bodies)
    }
    /// Screen: the same owners and anchors drafted by separate passes and by one batched pass, in the order
    /// separate, batched, batched, separate (twice; the first two trials warm up). Every trial's body outputs
    /// and private base logits are compared with the first separate trial's: bit equality, the largest
    /// absolute difference and the rows whose local argmax agrees. Nothing is accepted; positions must not move.
    pub(crate) fn probe_batched_body(nets: &mut [&mut QueryNetwork], anchors: &[u32], comm: &Roce, stream: &Stream)
        -> Result<serde_json::Value> {
        use serde_json::json;
        ensure!(crate::config::batched_draft_body() && (2..=3).contains(&nets.len()) && anchors.len() == nets.len(),
            "batched drafter screen extent");
        let bodies = nets[0].batch_bodies().context("batched drafter bodies are not prepared")?;
        for i in 1..nets.len() {
            ensure!(nets[i].shares_batch_bodies(&*nets[0]), "batched drafter screen owners of one origin");
        }
        let positions = nets.iter().map(|n| n.position()).collect::<Result<Vec<_>>>()?;
        let mut reference: Option<Vec<(Vec<bf16>, Vec<f32>)>> = None;
        let mut separate_stages: Option<Vec<Vec<(String, Vec<u16>)>>> = None;
        let mut first_difference: Option<Vec<serde_json::Value>> = None;
        let mut trials = Vec::new();
        for (trial, batched) in [false, true, true, false, false, true, true, false].into_iter().enumerate() {
            stream.synchronize()?;
            let began = std::time::Instant::now();
            if batched {
                bodies.get().context("batched drafter bodies")?.borrow_mut().run(nets, anchors, comm, stream)?;
            } else {
                for (net, &anchor) in nets.iter_mut().zip(anchors) { net.run(anchor, comm, stream)?; }
            }
            stream.synchronize()?;
            let milliseconds = began.elapsed().as_secs_f64() * 1000.;
            let outputs = nets.iter().map(|n| Ok((n.normalized.download()?, n.io.local_logits().download()?)))
                .collect::<Result<Vec<_>>>()?;
            ensure!(nets.iter().map(|n| n.position()).collect::<Result<Vec<_>>>()? == positions,
                "batched drafter screen advanced accepted state");
            let reference = reference.get_or_insert_with(|| outputs.clone());
            // Where the two passes first part: every stage buffer of each owner, in pass order.
            if !batched && separate_stages.is_none() {
                separate_stages = Some(nets.iter().map(|n| n.stage_snapshot()).collect::<Result<Vec<_>>>()?);
            }
            if batched && first_difference.is_none() {
                let cell = bodies.get().context("batched drafter bodies")?.borrow();
                let body = cell.bodies.iter().find(|b| b.rows == QUERY_ROWS * nets.len())
                    .context("batched drafter body")?;
                let mut found = Vec::new();
                for (k, net) in nets.iter().enumerate() {
                    let batch = body.stage_snapshot(k, net)?;
                    let separate = &separate_stages.as_ref().context("separate drafter stages")?[k];
                    ensure!(batch.len() == separate.len(), "drafter stage snapshot extents");
                    let mut first = serde_json::Value::Null;
                    let mut differing = 0usize;
                    for ((name, a), (other, b)) in separate.iter().zip(&batch) {
                        ensure!(name == other && a.len() == b.len() && a.len() % QUERY_ROWS == 0,
                            "drafter stage snapshot order");
                        if a == b { continue; }
                        differing += 1;
                        if first.is_null() {
                            let width = a.len() / QUERY_ROWS;
                            let rows: Vec<usize> = (0..QUERY_ROWS).map(|r| (0..width)
                                .filter(|i| a[r * width + i] != b[r * width + i]).count()).collect();
                            let largest = a.iter().zip(b).map(|(x, y)| (bf16::from_bits(*x).to_f32()
                                - bf16::from_bits(*y).to_f32()).abs()).fold(0f32, f32::max);
                            first = json!({"buffer":name,"differing_elements_per_row":rows,"max_abs":largest});
                        }
                    }
                    found.push(json!({"owner":k,"differing_buffers":differing,"of":separate.len(),"first":first}));
                }
                first_difference = Some(found);
            }
            let mut owners = Vec::new();
            for ((normalized, logits), (ref_normalized, ref_logits)) in outputs.iter().zip(reference.iter()) {
                ensure!(normalized.len() == ref_normalized.len() && logits.len() == ref_logits.len()
                    && logits.len() % QUERY_ROWS == 0, "batched drafter screen output extents");
                let exact_normalized = normalized.iter().zip(ref_normalized).all(|(a, b)| a.to_bits() == b.to_bits());
                let exact_logits = logits.iter().zip(ref_logits).all(|(a, b)| a.to_bits() == b.to_bits());
                let max_normalized = normalized.iter().zip(ref_normalized)
                    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0f32, f32::max);
                let max_logit = logits.iter().zip(ref_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let width = logits.len() / QUERY_ROWS;
                let argmax = |x: &[f32]| x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
                let argmax_rows = (0..QUERY_ROWS).filter(|r| argmax(&logits[r * width..(r + 1) * width])
                    == argmax(&ref_logits[r * width..(r + 1) * width])).count();
                owners.push(json!({"exact_normalized":exact_normalized,"exact_logits":exact_logits,
                    "max_abs_normalized":max_normalized,"max_abs_logit":max_logit,"argmax_equal_rows":argmax_rows}));
            }
            trials.push(json!({"trial":trial,"batched":batched,"warmup":trial < 2,"milliseconds":milliseconds,
                "owners":owners}));
        }
        let mean = |batched: bool| {
            let v: Vec<f64> = trials.iter().filter(|t| t["warmup"] == false && t["batched"] == batched)
                .filter_map(|t| t["milliseconds"].as_f64()).collect();
            v.iter().sum::<f64>() / v.len().max(1) as f64
        };
        let all_exact = trials.iter().all(|t| t["owners"].as_array().is_some_and(|o| o.iter()
            .all(|o| o["exact_normalized"] == true && o["exact_logits"] == true)));
        let best = |batched: bool| trials.iter().filter(|t| t["warmup"] == false && t["batched"] == batched)
            .filter_map(|t| t["milliseconds"].as_f64()).fold(f64::INFINITY, f64::min);
        Ok(json!({"rank":nets[0].rank,"owners":nets.len(),"positions":positions,"all_exact":all_exact,
            "separate_ms":mean(false),"batched_ms":mean(true),"separate_best_ms":best(false),"batched_best_ms":best(true),
            "first_difference":first_difference,"trials":trials,
            "scope":"Whole drafter passes (bodies and heads) of the same owners and anchors, separate against one batched body pass; synchronized wall time on this rank; Markov sampling excluded; accepted state unchanged."}))
    }
}
