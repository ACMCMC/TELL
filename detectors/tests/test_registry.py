from detectors_bench.registry import load_registry


def test_registry_contains_requested_detectors():
    registry = load_registry()
    expected = {
        "binoculars",
        "detectgpt",
        "fast_detectgpt",
        "mage_d",
        "openai_roberta",
        "argugpt",
        "detectllm_lrr",
        "detectllm_npr",
        "radar",
        "pangram_editlens_llama",
        "aigc_mpu_env3",
        "meld",
        "t5_sentinel",
        "dnagpt",
        "mfd",
        "logrank_gpt2_medium",
        "phd_roberta",
        "chatgpt_d",
        "ghostbuster",
    }
    assert expected <= set(registry)
