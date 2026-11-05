//! CPU-only protocol experiment: recipe stream -> real client -> next prompt.
//! No inference, request dispatch, or model-state modification.
use anyhow::{Result,ensure};
use canada_engine::protocol::{Frontend,MODEL_NAME};
use deepseek_recipe::{openai::ChatCompletionRequest,request::ProtocolRequest,
    stream::{InferenceChunk,InferenceFinishReason,PromptUsage,StreamProcessor}};
use serde_json::{Value,json};
use std::{path::Path,sync::Arc};
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;

fn main()->Result<()> {
    let a=std::env::args().collect::<Vec<_>>();
    ensure!(a.len()==5,"mode tokenizer input.json output.json");
    let frontend=Frontend::load(Path::new(&a[2]))?;
    let tokenizer=Arc::new(Tokenizer::from_file(&a[2]).map_err(anyhow::Error::msg)?);
    let mut cases:Vec<Value>=serde_json::from_slice(&std::fs::read(&a[3])?)?;
    ensure!(!cases.is_empty()&&cases.len()<=24,"bounded synthetic case count");
    let runtime=tokio::runtime::Builder::new_current_thread().build()?;
    for case in &mut cases {
        let first=frontend.prepare_chat(&serde_json::to_vec(&case["request"])?,1048576)?;
        if a[1]=="stream" {
            let mut ids=tokenizer.encode(case["output_text"].as_str().unwrap(),false)
                .map_err(anyhow::Error::msg)?.get_ids().to_vec();
            ids.push(1);ensure!(ids.len()<=first.max_new_tokens as usize,"output limit");
            let width=case["chunk_size"].as_u64().unwrap()as usize;
            ensure!((1..=64).contains(&width),"chunk width");
            let generator=ChatCompletionRequest::chunk_generator(&first.converted,
                "synthetic-roundtrip".into(),MODEL_NAME.into());
            let processor=StreamProcessor::new(generator,first.converted.parsing_options.clone())
                .with_tokenizer(tokenizer.clone());
            let mut events=vec![InferenceChunk::Ready{system_fingerprint:Some("cpu-protocol".into()),
                prompt_usage:PromptUsage{prompt_tokens:first.tokens.len(),prompt_cache_hit_tokens:0}}];
            events.extend(ids[..ids.len()-1].chunks(width).map(|v|InferenceChunk::Tokens{
                token_ids:v.to_vec(),logprobs:None}));
            events.push(InferenceChunk::Finish{finish_reason:InferenceFinishReason::Stop});
            let frames=runtime.block_on(async {
                let mut stream=std::pin::pin!(processor.into_protocol_stream(tokio_stream::iter(events)));
                let mut frames=Vec::new();
                while let Some(frame)=stream.next().await {frames.push(serde_json::to_value(frame?)?);}
                Ok::<_,anyhow::Error>(frames)
            })?;
            case["output_ids"]=json!(ids);case["frames"]=json!(frames);
        }else if a[1]=="compare" {
            let next=frontend.prepare_chat(&serde_json::to_vec(&case["next_request"])?,1048576)?;
            let ids:Vec<u32>=serde_json::from_value(case["output_ids"].clone())?;
            ensure!(ids.last()==Some(&1),"synthetic final EOS");
            let mut generated=first.tokens.clone();generated.extend_from_slice(&ids[..ids.len()-1]);
            let common=generated.iter().zip(&next.tokens).take_while(|(a,b)|a==b).count();
            let assistant=case["next_request"]["messages"].as_array().unwrap().iter()
                .find(|m|m["role"]=="assistant").cloned().unwrap();
            case["comparison"]=json!({"prompt_tokens":first.tokens.len(),"generated_tokens":ids.len(),
                "committed_tokens":generated.len(),"next_prompt_tokens":next.tokens.len(),
                "common_prefix_tokens":common,"exact_live_prefix":next.tokens.len()>generated.len()&&next.tokens.starts_with(&generated),
                "reasoning_returned":assistant["reasoning_content"].as_str().is_some_and(|s|!s.is_empty()),
                "generated_mismatch_tail":frontend.decode(&generated[common..])?,
                "next_mismatch_tail":frontend.decode(&next.tokens[common..])?});
            case.as_object_mut().unwrap().remove("frames");
        }else{anyhow::bail!("unknown mode");}
    }
    std::fs::write(&a[4],serde_json::to_vec_pretty(&cases)?)?;Ok(())
}
