//! Response metadata from measurements the model owner already records.
//! No clocks, CUDA calls, synchronization or inference-policy changes.
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct PrefillTiming {
    pub prompt_n: usize,
    pub cache_n: usize,
    pub prompt_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    pub scope: &'static str,
}

impl PrefillTiming {
    pub fn from_worker(event: &Value, prompt_tokens: usize) -> Option<Self> {
        // A partial background prefill has not measured the whole suffix yet.
        if event["type"] != "tokens" || event["prompt_tokens"].as_u64()? != prompt_tokens as u64 {
            return None;
        }
        let cache_n = usize::try_from(event["cached_tokens"].as_u64()?).ok()?;
        let prompt_n = prompt_tokens.checked_sub(cache_n)?;
        let seconds = event["begin_phases_seconds"]["target_prefill"].as_f64()?;
        if !seconds.is_finite() || seconds < 0. || (prompt_n > 0 && seconds == 0.) {
            return None;
        }
        let prompt_ms = seconds * 1000.;
        let prompt_per_second = (prompt_n > 0).then(|| prompt_n as f64 / seconds);
        if !prompt_ms.is_finite() || prompt_per_second.is_some_and(|rate| !rate.is_finite()) {
            return None;
        }
        Some(Self {
            prompt_n, cache_n, prompt_ms, prompt_per_second,
            scope: if event["background_prefill"] == true {
                "server_prefill_active"
            } else {
                "server_prefill_phase"
            },
        })
    }
}

/// Extend the wire object without re-parsing, cloning or modifying its recipe
/// fields. Ordinary token chunks continue through the original serializer.
#[derive(Serialize)]
pub(super) struct WithTimings<'a, T> {
    #[serde(flatten)]
    pub response: T,
    pub timings: &'a PrefillTiming,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(cached: usize, seconds: f64) -> Value {
        json!({"type":"tokens","prompt_tokens":10000,"cached_tokens":cached,
            "seconds":100.,"queue_and_prefill_seconds":150.,
            "begin_phases_seconds":{"target_prefill":seconds}})
    }

    #[test]
    fn fresh_and_cached_rates_use_measured_phase_and_only_uncached_tokens() {
        for (cached, expected) in [(0, 20000.), (9900, 200.)] {
            let timing = PrefillTiming::from_worker(&event(cached, 0.5), 10000).unwrap();
            assert_eq!(timing.prompt_n, 10000 - cached);
            assert_eq!(timing.cache_n, cached);
            assert_eq!(timing.prompt_ms, 500.);
            assert_eq!(timing.prompt_per_second, Some(expected));
            assert_eq!(timing.scope, "server_prefill_phase");
        }
    }

    #[test]
    fn background_completion_reports_accumulated_active_phase_only() {
        let mut input = event(9900, 0.25);
        input["background_prefill"] = json!(true);
        let timing = PrefillTiming::from_worker(&input, 10000).unwrap();
        assert_eq!(timing.scope, "server_prefill_active");
        assert_eq!(timing.prompt_per_second, Some(400.));
        input["type"] = json!("prefill_progress");
        assert!(PrefillTiming::from_worker(&input, 10000).is_none());
    }

    #[test]
    fn missing_invalid_or_zero_work_cannot_invent_a_prefill_rate() {
        for input in [event(10001, 0.5), event(0, -1.), event(0, 0.), event(0, f64::NAN),
            event(0, f64::MAX), event(0, f64::MIN_POSITIVE / 2.),
            json!({"type":"tokens","prompt_tokens":10000,"cached_tokens":0,"seconds":1.})] {
            assert!(PrefillTiming::from_worker(&input, 10000).is_none());
        }
        assert!(PrefillTiming::from_worker(&event(0, 1.), 9999).is_none());
        let zero = PrefillTiming::from_worker(&event(10000, 0.), 10000).unwrap();
        assert_eq!(zero.prompt_n, 0);
        assert!(zero.prompt_per_second.is_none());
    }

    #[test]
    fn wrapper_preserves_all_existing_protocol_fields() {
        let response = json!({"id":"example","choices":[{"delta":{"reasoning_content":"text",
            "tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}],
            "usage":{"prompt_tokens":10000,"completion_tokens":4}});
        let timing = PrefillTiming::from_worker(&event(9900, 0.5), 10000).unwrap();
        let mut encoded = serde_json::to_value(WithTimings { response: &response, timings: &timing }).unwrap();
        assert_eq!(encoded.as_object_mut().unwrap().remove("timings").unwrap(), serde_json::to_value(timing).unwrap());
        assert_eq!(encoded, response);
    }
}
