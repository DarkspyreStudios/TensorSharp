// DeepSeek V4.1 visual spans in the whole-model text executor.
// Included after TSGgml_Dsv4Forward; the caller serializes access to a model.

TSG_EXPORT int TSGgml_Dsv41AttachVision(void * text_handle, void * vision_handle)
{
    if (!text_handle || !vision_handle) return -1;
    auto & m = *static_cast<tsg_dsv4::dsv4_model *>(text_handle);
    try
    {
        auto encoder = tsg_dsv41_vision::from_handle(vision_handle);
        if (!m.hp.v41 || !encoder || m.vision)
            throw std::runtime_error("V4.1 vision requires a V4.1 model without an existing companion");
        const auto & p = encoder->params();
        if (p.text_dim != m.hp.n_embd || p.text_layers != m.hp.n_layer ||
            p.n_experts != m.hp.n_expert || p.tokenizer_hash != m.engram.tokenizer_hash ||
            p.image_token_id < 0 || p.image_token_id >= m.hp.n_vocab)
            throw std::runtime_error("V4.1 vision companion does not match the text model and tokenizer");

        auto attachment = std::make_unique<tsg_dsv4::dsv41_vision_attachment>();
        attachment->encoder = std::move(encoder);
        attachment->router_bias.resize(m.hp.n_layer);
        // Keep the two trained biases as exact rows. get_rows selects either
        // row per position, avoiding cancellation from a (visual - text) blend.
        for (int il = 0; il < m.hp.n_layer; ++il)
        {
            const int dev = m.layers[il].device;
            if (!attachment->ctx[dev])
            {
                ggml_init_params init = {
                    (size_t) (m.hp.n_layer + 4) * ggml_tensor_overhead() + 4096, nullptr, true };
                attachment->ctx[dev] = ggml_init(init);
                if (!attachment->ctx[dev]) throw std::runtime_error("V4.1 vision bias context allocation failed");
            }
            auto * bias = ggml_new_tensor_2d(attachment->ctx[dev], GGML_TYPE_F32, m.hp.n_expert, 2);
            ggml_format_name(bias, "blk.%d.ffn_router_text_vision_bias", il);
            attachment->router_bias[il] = bias;
            const auto & visual = attachment->encoder->router_bias(il);
            if (visual.size() != (size_t) m.hp.n_expert ||
                !std::all_of(visual.begin(), visual.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Invalid V4.1 visual routing bias");
        }
        for (int dev = 0; dev <= m.n_gpu; ++dev)
        {
            if (!attachment->ctx[dev]) continue;
            attachment->buf[dev] = ggml_backend_alloc_ctx_tensors(attachment->ctx[dev], m.backends[dev]);
            if (!attachment->buf[dev]) throw std::runtime_error("V4.1 vision bias buffer allocation failed");
            ggml_backend_buffer_set_usage(attachment->buf[dev], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        std::vector<float> values((size_t) m.hp.n_expert * 2);
        for (int il = 0; il < m.hp.n_layer; ++il)
        {
            auto * text_bias = m.layers[il].ffn_exp_probs_b;
            if (!text_bias || ggml_nelements(text_bias) != m.hp.n_expert)
                throw std::runtime_error("Invalid V4.1 text routing bias");
            std::vector<uint8_t> bytes(ggml_nbytes(text_bias));
            ggml_backend_tensor_get(text_bias, bytes.data(), 0, bytes.size());
            if (text_bias->type == GGML_TYPE_F32)
                memcpy(values.data(), bytes.data(), (size_t) m.hp.n_expert * sizeof(float));
            else if (auto convert = ggml_get_type_traits(text_bias->type)->to_float)
                convert(bytes.data(), values.data(), m.hp.n_expert);
            else throw std::runtime_error("Unsupported V4.1 text routing bias type");
            const auto & visual = attachment->encoder->router_bias(il);
            std::copy(visual.begin(), visual.end(), values.begin() + m.hp.n_expert);
            ggml_backend_tensor_set(attachment->router_bias[il], values.data(), 0, values.size() * sizeof(float));
        }
        m.vision = std::move(attachment);
        return 0;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv41] vision attachment failed: %s\n", e.what());
        return -2;
    }
}

TSG_EXPORT int TSGgml_Dsv41ForwardVision(void * handle, const int32_t * tokens,
    const uint8_t * image_mask, const float * image_embeddings,
    int n_tokens, int n_image_tokens, float * logits_out)
{
    if (!handle || !tokens || !image_mask || !logits_out || n_tokens <= 0 ||
        n_image_tokens < 0 || n_image_tokens > n_tokens || (n_image_tokens && !image_embeddings)) return -1;
    auto & m = *static_cast<tsg_dsv4::dsv4_model *>(handle);
    auto & slot = *m.active_slot;
    if (m.hp.v41 && slot.v41_failed)
    {
        fprintf(stderr, "[dsv41] sequence execution previously failed; reset or destroy the slot before reuse\n");
        return -4;
    }
    bool execution_started = false;
    try
    {
        if (!m.hp.v41 || (n_image_tokens && !m.vision))
            throw std::runtime_error("Visual inference requires an attached V4.1 vision companion");
        if (slot.n_past < 0 || n_tokens > m.n_ctx - slot.n_past)
            throw std::runtime_error("V4.1 visual prompt exceeds the context limit");
        int count = 0;
        for (int i = 0; i < n_tokens; ++i)
        {
            if (image_mask[i] > 1 || tokens[i] < 0 || tokens[i] >= m.hp.n_vocab)
                throw std::runtime_error("Invalid V4.1 image mask or token ID");
            if (image_mask[i])
            {
                if (!m.vision || tokens[i] != m.vision->encoder->params().image_token_id)
                    throw std::runtime_error("Visual positions must use the checkpoint's image token ID");
                ++count;
            }
        }
        if (count != n_image_tokens)
            throw std::runtime_error("V4.1 compact image row count differs from its position mask");
        const size_t values = (size_t) n_image_tokens * m.hp.n_embd;
        for (size_t i = 0; i < values; ++i)
            if (!std::isfinite(image_embeddings[i]))
                throw std::runtime_error("V4.1 image embeddings must be finite");
        if (!count) return TSGgml_Dsv4Forward(handle, tokens, n_tokens, logits_out);

        if (m.moe_tp) m.moe_tp->begin_forward();
        execution_started = true;
        m.pos_end_hint = (int64_t) slot.n_past + n_tokens;
        int done = 0, image_done = 0;
        while (done < n_tokens)
        {
            const int nt = std::min(m.n_ubatch, n_tokens - done);
            int images = 0;
            for (int i = 0; i < nt; ++i) images += image_mask[done + i];
            const bool last = done + nt == n_tokens;
            if (!tsg_dsv4::dsv4_forward_ubatch(m, tokens + done, nt, slot.n_past,
                last, last ? logits_out : nullptr, false, image_mask + done,
                image_embeddings + (size_t) image_done * m.hp.n_embd, images))
            {
                slot.v41_failed = true;
                return -3;
            }
            slot.n_past += nt;
            done += nt;
            image_done += images;
        }
        // Same prompt-boundary rule as the text forward: this call is a prompt.
        if (n_tokens > 1) tsg_dsv4::dsv4_checkpoint_slot(m, slot);
        return 0;
    }
    catch (const std::exception & e)
    {
        if (execution_started) slot.v41_failed = true;
        fprintf(stderr, "[dsv41] visual forward failed: %s\n", e.what());
        return -2;
    }
    catch (...)
    {
        if (execution_started) slot.v41_failed = true;
        fprintf(stderr, "[dsv41] visual forward failed with an unknown execution exception\n");
        return -2;
    }
}
