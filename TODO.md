# Task 1
```
def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs
```
When batch size is greater than 1, I want to be able to also add the ability to provide the exact input id lengths as part of the input.

# Task 2
I want to programmatically call the sglang bench latency in python and get the prefill/decode times.

# Task 3
I want to run a benchmark that records different batch sizes of prefill. Different batch sizes of decode and the relevant execution time

# Task 4
I want to run a benchmark that collects decode execution time when the bs is skewed. Possibly in a heavy tailed fashion vs an even fashion