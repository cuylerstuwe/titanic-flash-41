//! Finite foreground policy experiment. CPU arithmetic is timing/count
//! bookkeeping only; all target/drafter probability arithmetic stays on GPU.
use super::*;

struct Feedback {
    hot:bool, rounds:usize, tokens:u64, seconds:f64, windows:usize,
    rates:[Option<f64>;2],
}
impl Feedback {
    fn new(hot:bool)->Self {
        Self{hot,rounds:0,tokens:0,seconds:0.,windows:0,rates:[None,None]}
    }
    fn observe(&mut self,tokens:u64,seconds:f64)->Option<bool> {
        self.rounds+=1;self.tokens+=tokens;self.seconds+=seconds;
        if self.rounds<8 {return None;}
        let arm=usize::from(self.hot);let rate=self.tokens as f64/self.seconds;
        self.rates[arm]=Some(self.rates[arm].map_or(rate,|old|0.5*old+0.5*rate));
        self.rounds=0;self.tokens=0;self.seconds=0.;self.windows+=1;
        let other=1-arm;
        let next=if self.rates[other].is_none()||self.windows%4==0 {other}
            else if self.rates[other].unwrap()>self.rates[arm].unwrap()*1.03 {other}
            else{arm};
        Some(next==1)
    }
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_SOURCE_PRIOR_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_ROOT_TEMPERATURE_SCREEN").as_deref()==Ok("1")
        &&crate::layer_attention::generation_fp32_attention_enabled()?
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&w.cached_sessions.is_empty()&&!w.prompt_checkpoint&&!w.prompt_cache_logits,
        "source-prior finite foreground admission");
    let transfer=std::env::var("CANADA_SCRATCH_SOURCE_PRIOR_TRANSFER").as_deref()==Ok("1");
    let raw:&[u8]=if transfer{include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json")}
        else{include_bytes!("../tests/fixtures/source-prior-exploration-tokens-v1.json")};
    let fixture:Value=serde_json::from_slice(raw)?;
    ensure!(fixture["thinking_mode"]=="thinking"&&fixture["reasoning_effort"]=="high",
        "source-prior canonical reasoning template");
    let cases=fixture["cases"].as_array().unwrap();ensure!(cases.len()==4,"source-prior case count");
    let mut id=1u64;let mut records=Vec::new();let mut append_check=None;
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true)
        .open(out.join("source-prior-records.jsonl"))?;
    for (ordinal,case) in cases.iter().enumerate() {
        let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        let prose=case["category"]=="prose";
        ensure!((128..=512).contains(&prompt.len())&&prompt.last()==Some(&128821)
            &&(prose||case["category"]=="code"),"source-prior prompt identity");
        let seed=if transfer{202680800}else{202680200}+ordinal as u64;
        // Excluded warmups prepare both actual-q root schedules. No GPU
        // overlap estimator or external classifier runs in this experiment.
        if ordinal==0 {
            for hot in [false,true] {
                w.execute(Command::Begin{images: Vec::new(),id,seed,greedy:false,limit:64,tokens:prompt.clone()})?;id+=1;
                w.session.as_mut().unwrap().set_root_temperature_for_probe(hot,&w.stream)?;
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id){w.execute(Command::Step)?;}
                w.session.as_mut().unwrap().set_root_temperature_for_probe(false,&w.stream)?;
                w.execute(Command::Finish(if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length}))?;
            }
        }
        // Code's specified prior is cold, identical to feedback-only. Avoid
        // executing that redundant third arm. Rotate order across fresh tasks.
        let policies:Vec<u8>=if transfer{if ordinal%2==0{vec![0,1]}else{vec![1,0]}}
            else{match ordinal {0=>vec![0,1],1=>vec![1,0],2=>vec![2,0,1],_=>vec![1,0,2]}};
        for policy in policies {
            let begin=w.execute(Command::Begin{images: Vec::new(),id,seed,greedy:false,limit:384,tokens:prompt.clone()})?;id+=1;
            ensure!(begin["generation_fp32_attention"]==true&&begin["target_temperature"]==1.
                &&begin["target_top_p"]==1.&&begin["block_verification"]==true,
                "source-prior target policy must remain fixed");
            let mut feedback=Feedback::new(policy==2&&prose);
            w.session.as_mut().unwrap().set_root_temperature_for_probe(feedback.hot,&w.stream)?;
            let mut output:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let mut steps=Vec::new();let mut decisions=Vec::new();let mut answer=false;
            w.control.agree(b"source-prior-measure")?;let started=Instant::now();
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                let at=Instant::now();let step=w.execute(Command::Step)?;let seconds=at.elapsed().as_secs_f64();
                let emitted:Vec<u32>=serde_json::from_value(step["output_ids"].clone())?;
                let ended=!answer&&emitted.contains(&128822);output.extend_from_slice(&emitted);
                let previously_hot=feedback.hot;
                let mut next=None;
                if ended {
                    answer=true;feedback=Feedback::new(false);next=Some(false);
                }else if policy!=0&&step["proposal_source"]=="dspark"&&step["step_limit"].as_u64().unwrap()>=6 {
                    next=feedback.observe(step["committed_model_rows"].as_u64().unwrap(),seconds);
                }
                if let Some(proposed)=next {
                    // Rank0's observed end-to-end step cost selects the policy;
                    // every rank receives the same choice before the next draft.
                    let byte=[u8::from(proposed)];
                    let wire=w.control.command((rank==0).then_some(byte.as_slice()))?;
                    ensure!(wire.len()==1&&wire[0]<=1,"source-prior decision wire");
                    let hot=wire[0]==1;
                    decisions.push(json!({"position":w.verify.position(),"answer":answer,
                        "old_hot":previously_hot,"next_hot":hot,"window":feedback.windows,
                        "local_rates":feedback.rates,"phase_reset":ended}));
                    if hot!=previously_hot||ended {
                        w.session.as_mut().unwrap().set_root_temperature_for_probe(hot,&w.stream)?;
                    }
                    feedback.hot=hot;
                }
                steps.push(step);
            }
            let seconds=started.elapsed().as_secs_f64();
            w.session.as_mut().unwrap().set_root_temperature_for_probe(false,&w.stream)?;
            let stop=w.history.pending()==Some(w.cfg.eos_token_id);
            let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
            ensure!(w.history.frontier()==prompt.len()+output.len()-1&&w.history.pending()==output.last().copied()
                &&output.len()==w.emitted&&(stop||w.emitted==384),"source-prior committed frontier");
            let record=json!({"case":case["id"],"category":case["category"],"family":case["family"],
                "policy":policy,"seed":seed,"prompt_tokens":prompt.len(),"output_ids":output,
                "decode_seconds":seconds,"begin":begin,"finish":finish,"steps":steps,"decisions":decisions,
                "answer_phase_reached":answer,"uniform_policy_restored":true});
            writeln!(progress,"{}",record)?;progress.flush()?;records.push(record);
            // One short retained append also exercises the new generation
            // primitive's transition to the existing BF16 append bank and back.
            if ordinal==0&&policy==0 {
                let appended=[w.history.committed(),&prompt[prompt.len()-6..]].concat();
                let b=w.execute(Command::Begin{images: Vec::new(),id,seed:seed+100,greedy:false,limit:16,tokens:appended.clone()})?;id+=1;
                ensure!(b["uncached_tokens"]==6&&b["generation_fp32_attention"]==true,
                    "source-prior short append missed retained route");
                let mut ids:Vec<u32>=serde_json::from_value(b["output_ids"].clone())?;
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id){
                    let s=w.execute(Command::Step)?;ids.extend(serde_json::from_value::<Vec<u32>>(s["output_ids"].clone())?);
                }
                let f=w.execute(Command::Finish(if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length}))?;
                ensure!(w.history.frontier()==appended.len()+ids.len()-1,"short append generation frontier");
                append_check=Some(json!({"begin":b,"finish":f,"output_ids":ids,"frontier_checked":true}));
            }
            eprintln!("source prior rank{rank} {} policy{policy}",case["id"]);
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("source-prior-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"append_check":append_check,"startup_warmup":startup,"transfer":transfer,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"all_committed_frontiers_checked":true,
        "target_temperature":1.,"target_top_p":1.,"generation_fp32_attention":true,"serving_policy_changed":false,
        "scope":if transfer{"Four previously unused task families. Same s803 feedback rule, unchanged8-step windows,EMA0.5,3% hysteresis and fourth-window exploration. Fixed versus feedback only, one seed per task and384-token caps. No category prior or classifier; actual-q correction unchanged. One16-token append check. Transfer screen, not completed quality outcomes."}else{"Four fresh short prompts: two coding, two prose. One seed per task;384-token caps. Fixed T.7 versus8-neural-step window feedback using committed tokens/time; prose adds initial rootT1 prior while reasoning, then same feedback. Suffix drafts remainT.7; actual-q correction unchanged. Oracle source labels, no classifier. Code prior equals feedback-only, so no duplicate arm. One16-token short-append integration check. Initial hypothesis screen, not completed outcomes or generalization proof."}}))?)?;
    Ok(())
}
