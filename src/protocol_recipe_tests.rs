//! Integration boundaries selected from official deepseek-recipe 0.2.0.
//! Compare our optimized/cached tokenizer with the official uncached encoder.
//! No inference, model math, or backend requests are involved.
use super::*;

fn frontend() -> Frontend {
    Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap()
}

fn tool_request() -> Value {
    json!({"max_tokens":128,"messages":[
        {"role":"user","content":"Read both files. Literal <think> and </think>."},
        {"role":"assistant","content":"","reasoning_content":"Read each file independently.",
         "tool_calls":[
            {"id":"call_a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.rs\"}"}},
            {"id":"call_b","type":"function","function":{"name":"read","arguments":"{\"path\":\"b.rs\"}"}}]},
        {"role":"tool","tool_call_id":"call_a","content":"fn a() {}"},
        {"role":"tool","tool_call_id":"call_b","content":"fn b() {}"}],
        "tools":[{"type":"function","function":{"name":"read","parameters":{
            "type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]})
}

fn compare_paths(f: &Frontend, body: &Value) -> PreparedChat {
    let bytes=serde_json::to_vec(body).unwrap();
    let official=f.prepare_chat_impl(&bytes,8192,0).unwrap();
    for mode in [1,2,2] {
        let optimized=f.prepare_chat_impl(&bytes,8192,mode).unwrap();
        assert_eq!(optimized.tokens,official.tokens,"tokenizer mode {mode}");
        assert_eq!(optimized.rendered_sha256,official.rendered_sha256);
    }
    official
}

#[test]
#[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
fn recipe_contract_tool_order_and_effort_preserve_canonical_encoding() {
    let f=frontend();
    let mut request=tool_request();
    let mut high_tokens=None;
    for (effort,score) in [("high",75),("max",100)] {
        request["reasoning_effort"]=json!(effort);
        let ordered=compare_paths(&f,&request);
        let text=f.decode(&ordered.tokens).unwrap();
        assert!(text.contains(&format!("Reasoning Effort: {score} (range 1-100")));
        assert!(text.contains("Read each file independently."));
        let mut reversed=request.clone();
        reversed["messages"].as_array_mut().unwrap().swap(2,3);
        assert_eq!(ordered.tokens,compare_paths(&f,&reversed).tokens,
            "tool result arrival order must normalize to call order");
        if let Some(high)=high_tokens.as_ref() { assert_ne!(&ordered.tokens,high); }
        else { high_tokens=Some(ordered.tokens); }
    }
}

#[test]
#[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
fn recipe_contract_prefix_and_adjacent_users_keep_cache_boundaries() {
    let f=frontend();
    let separate=json!({"max_tokens":32,"messages":[
        {"role":"user","content":"First 😀"},{"role":"user","content":"Second 中文"}]});
    let merged=json!({"max_tokens":32,"messages":[{"role":"user","content":"First 😀\n\nSecond 中文"}]});
    assert_eq!(compare_paths(&f,&separate).tokens,compare_paths(&f,&merged).tokens);
    for (content,reasoning,suffix) in [
        ("Draft answer: ","Prepared reasoning.","Draft answer: "),
        ("","Continue reasoning here","Continue reasoning here"),
    ] {
        let request=json!({"max_tokens":32,"messages":[{"role":"user","content":"Continue."},
            {"role":"assistant","content":content,"reasoning_content":reasoning,"prefix":true}]});
        let actual=compare_paths(&f,&request);
        let text=f.decode(&actual.tokens).unwrap();
        assert!(text.ends_with(suffix),"unexpected assistant continuation suffix: {text:?}");
    }
}

#[test]
#[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
fn recipe_contract_parallel_tool_stream_preserves_typed_arguments() {
    let f=frontend();
    let request=tool_request();
    let prepared=compare_paths(&f,&request);
    let text="Choose both.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">目录/😀.rs</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">b.rs</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"options\" string=\"false\">{\"lines\": [1, 3], \"raw\": true, \"end\": null}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let mut tokens=f.tokenizer.encode(text,false).unwrap().get_ids().to_vec();
    tokens.push(1);
    let rt=tokio::runtime::Builder::new_current_thread().build().unwrap();
    for width in [1,3,17,256] {
        let response=rt.block_on(f.parse_completion(&prepared,&tokens,"stop",width)).unwrap();
        let choice=&response["choices"][0];
        assert_eq!(choice["finish_reason"],"tool_calls");
        assert_eq!(choice["message"]["reasoning_content"],"Choose both.");
        let calls=choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(),2);
        assert_ne!(calls[0]["id"],calls[1]["id"]);
        let args:Vec<Value>=calls.iter().map(|call| {
            assert_eq!(call["function"]["name"],"read");
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
        }).collect();
        assert_eq!(args,vec![json!({"path":"目录/😀.rs"}),
            json!({"path":"b.rs","options":{"lines":[1,3],"raw":true,"end":null}})]);
        assert_eq!(response["usage"]["completion_tokens"],tokens.len()-1);
    }
}
