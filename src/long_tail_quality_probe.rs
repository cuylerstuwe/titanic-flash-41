//! A small exact-history long-context logit export after serving shuts down.
//! All probability/NLL calculations remain in the subsequent GPU diagnostic.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::path::Path;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_POST_HTTP_LONG_QUALITY").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && verify.physical_rows()==6
        && verify.head_gemv_enabled() && verify.fused_compact_publication_layers()==4
        && verify.engram_input_kind()=="mapped" && verify.paired_engram_io_enabled()
        && wides.len()==5,"long-tail quality requires the accumulated serving candidate");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture_sha=format!("{:x}",Sha256::digest(raw));
    ensure!(fixture_sha=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb",
        "long-tail corpus identity");
    let fixture:Value=serde_json::from_slice(raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let directory=output.file_name().and_then(|s|s.to_str()).unwrap_or("");
    let prefix=format!("target-native-r{rank}-");
    let version=directory.strip_prefix(&prefix).ok_or_else(||anyhow::anyhow!("quality output identity"))?;
    ensure!(version.starts_with('s') && version[1..].bytes().all(|b|b.is_ascii_digit()),"quality version");
    let raw_root=output.parent().unwrap().join(format!("quality-long-tail-{version}"));
    if rank==3 {std::fs::create_dir(&raw_root)?;}
    let mut records=Vec::new();let mut frontiers=Vec::new();
    for (index,frontier) in [131072,262144].into_iter().enumerate() {
        verify.reset(700000+index as u64,comm,stream)?;large.handoff_from(verify)?;
        large.prefill_wide(wides,&corpus[..frontier],comm,stream)?;
        verify.handoff_from(large)?;
        for offset in (0..64).step_by(6) {
            let live=(64-offset).min(6);let position=frontier+offset;
            ensure!(verify.position()==position as u64,"quality accepted history frontier");
            verify.begin_verification(&corpus[position..position+live],comm,stream)?;
            let (logits,rows)=verify.pending_logits()?;ensure!(rows==live,"quality live logits");
            stream.synchronize()?;
            let bytes=unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?;
            for row in 0..live {
                let values=&bytes[row*129280*4..(row+1)*129280*4];
                let file=format!("f{frontier}-r{}.bin",offset+row);
                if rank==3 {std::fs::write(raw_root.join(&file),values)?;}
                records.push(json!({"frontier":frontier,"row":offset+row,"position":position+row,
                    "input_id":corpus[position+row],"expected_next_id":corpus[position+row+1],
                    "bytes":values.len(),"sha256":format!("{:x}",Sha256::digest(values)),
                    "file":file,"rank":rank}));
            }
            verify.finish_verification(live,comm,stream)?;
        }
        ensure!(verify.position()==(frontier+64)as u64,"quality committed suffix");
        frontiers.push(json!({"prefix_tokens":frontier,"final_position":verify.position(),
            "carry":verify.carry_digests()?,"encoder":verify.encoder_tail_digest()?}));
        eprintln!("long-tail quality rank{rank} frontier{frontier}: exported64rows");
    }
    let result=json!({"rank":rank,"fixture_sha256":fixture_sha,"records":records,"frontiers":frontiers,
        "raw_directory":raw_root,"raw_rank":3,"target_head":"canonical_bf16_weights_fp32_output",
        "fused_compact_publication_layers":verify.fused_compact_publication_layers(),
        "performance_result":false,"quality_parity_proven":false,
        "scope":"Two64-token public-corpus windows after131072/262144 accepted tokens. Exact teacher-forced histories in the accumulated native serving target, exported after HTTP shutdown. GPU NLL analysis pending; compare SGLang input-logprob histories with prefill-versus-verification arithmetic explicit. Small diagnostic proxy, not task quality."});
    if rank==3 {std::fs::write(raw_root.join("index.json"),serde_json::to_vec_pretty(&result)?)?;}
    Ok(result)
}
