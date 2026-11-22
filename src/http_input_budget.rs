//! Size the blocking JSON conversion separately from image pixel preparation.
//! Count only inline data in a real message image part. Text, schemas, tool
//! arguments, unknown shapes and escaped URLs keep the full text allowance.
use serde::Deserialize;
use serde::de::{Deserializer, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::{fmt, marker::PhantomData};

pub(super) const LARGE_BODY: usize = 16 << 20;

pub(super) fn workspace(total: usize, image_bytes: usize) -> usize {
    let images = image_bytes.min(total);
    // Text may be expanded by rendering and tokenization. Encoded image strings
    // can be held by JSON, Recipe, rendered sources and deferred image owners,
    // but never enter the text tokenizer. Pixel preparation is charged separately.
    (total - images) * 16 + images * 8 + (64 << 20)
}

fn sum_array<'de, D, T>(deserializer: D) -> Result<usize, D::Error>
where D: Deserializer<'de>, T: Deserialize<'de> + Into<usize> {
    struct Sum<T>(PhantomData<T>);
    impl<'de, T: Deserialize<'de> + Into<usize>> Visitor<'de> for Sum<T> {
        type Value = usize;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("an array") }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<usize, A::Error> {
            let mut bytes = 0;
            while let Some(item) = seq.next_element::<T>()? { bytes += item.into(); }
            Ok(bytes)
        }
    }
    deserializer.deserialize_seq(Sum(PhantomData::<T>))
}

#[derive(Deserialize)]
struct Part<'a> {
    #[serde(default, rename="type")]
    kind: &'a str,
    #[serde(borrow)]
    image_url: Option<&'a RawValue>,
}
#[derive(Deserialize)]
struct ImageUrl<'a> { url: &'a str }
impl From<Part<'_>> for usize {
    fn from(part: Part<'_>) -> usize {
        if part.kind != "image_url" { return 0; }
        let Some(url) = part.image_url.and_then(|raw| serde_json::from_str::<ImageUrl>(raw.get()).ok()) else { return 0; };
        let Some(data) = url.url.strip_prefix("data:image/") else { return 0; };
        data.split_once(";base64,").filter(|(mime,_)| matches!(*mime,"jpeg"|"png"|"webp"|"gif"))
            .map_or(0, |(_,payload)| payload.len())
    }
}
#[derive(Deserialize)]
struct Message<'a> { role: &'a str, #[serde(borrow)] content: Option<&'a RawValue> }
impl From<Message<'_>> for usize {
    fn from(message: Message<'_>) -> usize {
        if !matches!(message.role,"system"|"user"|"tool") { return 0; }
        let Some(parts) = message.content.filter(|v| v.get().starts_with('[')) else { return 0; };
        sum_array::<_,Part>(&mut serde_json::Deserializer::from_str(parts.get())).unwrap_or(0)
    }
}
#[derive(Deserialize)]
struct Request {
    #[serde(deserialize_with="sum_array::<_, Message>")]
    messages: usize,
}

pub(super) fn request_images(bytes: &[u8]) -> usize {
    serde_json::from_slice::<Request>(bytes).map_or(0, |r| r.messages)
}
pub(super) fn message_images(message: &str) -> usize {
    serde_json::from_str::<Message>(message).map_or(0, usize::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn only_unescaped_inline_image_parts_receive_image_accounting() {
        let url = "data:image/png;base64,QUJD";
        let image = json!({"type":"image_url","image_url":{"url":url}});
        let msg = json!({"role":"user","content":[image,{"type":"text","text":url,"image_url":{"url":url}}]});
        let body = json!({"tools":[image],"messages":[msg,{"role":"tool","content":url},{"role":"assistant","content":[image]}]});
        let raw = serde_json::to_vec(&body).unwrap();
        assert_eq!(request_images(&raw),4);
        assert_eq!(message_images(&msg.to_string()),4);
        assert_eq!(workspace(raw.len(),4),workspace(raw.len(),0)-32);
        assert_eq!(request_images(body.to_string().replace("QUJD","QUJ\\u0044").as_bytes()),0);
        assert_eq!(request_images(br#"{"messages":[{"role":"user","content":"unterminated"#),0);
    }
}
