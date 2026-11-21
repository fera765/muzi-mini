package com.muzi.mini.data

/**
 * Catalogo de modelos baixaveis. O APP (no celular, com internet normal)
 * baixa esses arquivos direto do HuggingFace. Cada modelo e uma pasta de
 * arquivos ONNX Runtime GenAI (model.onnx + .data + genai_config + tokenizer).
 *
 * Nota de contexto: 1M de contexto REAL so vale para modelos de nucleo SSM.
 * Os modelos Transformer abaixo tem contexto util limitado pela RAM (o KV
 * cache cresce). O "muzi-500m-ssm" e o nosso alvo que destrava 1M de verdade.
 */
data class ModelFile(val name: String, val url: String, val sizeMB: Int)

data class ModelSpec(
    val id: String,
    val label: String,
    val paramsB: String,
    val approxRamMB: Int,
    val contextTokens: Int,
    val ssm: Boolean,                 // true = 1M real; false = limitado por RAM
    val files: List<ModelFile>
) {
    val totalMB get() = files.sumOf { it.sizeMB }
}

object ModelCatalog {
    // base de URL do HuggingFace (resolve = download direto do arquivo)
    private fun hf(repo: String, path: String) =
        "https://huggingface.co/$repo/resolve/main/$path"

    val models: List<ModelSpec> = listOf(
        ModelSpec(
            id = "qwen2.5-0.5b-q8",
            label = "Qwen2.5 0.5B Instruct (int8) — leve e rapido",
            paramsB = "0.5B", approxRamMB = 900, contextTokens = 32768, ssm = false,
            files = listOf(
                // model_quantized.onnx = pesos int8, KV em fp32 (compativel com o loop)
                ModelFile("model.onnx", hf("onnx-community/Qwen2.5-0.5B-Instruct", "onnx/model_quantized.onnx"), 512),
                ModelFile("tokenizer.json", hf("onnx-community/Qwen2.5-0.5B-Instruct", "tokenizer.json"), 3),
                ModelFile("config.json", hf("onnx-community/Qwen2.5-0.5B-Instruct", "config.json"), 1)
            )
        ),
        ModelSpec(
            id = "qwen2.5-1.5b-q8",
            label = "Qwen2.5 1.5B Instruct (int8) — qualidade alta (professor)",
            paramsB = "1.5B", approxRamMB = 2000, contextTokens = 32768, ssm = false,
            files = listOf(
                ModelFile("model.onnx", hf("onnx-community/Qwen2.5-1.5B-Instruct", "onnx/model_quantized.onnx"), 1600),
                ModelFile("tokenizer.json", hf("onnx-community/Qwen2.5-1.5B-Instruct", "tokenizer.json"), 3),
                ModelFile("config.json", hf("onnx-community/Qwen2.5-1.5B-Instruct", "config.json"), 1)
            )
        ),
        ModelSpec(
            id = "muzi-500m-ssm",
            label = "Muzi 500M SSM — nosso aluno (1M contexto real) [em treino]",
            paramsB = "0.5B", approxRamMB = 234, contextTokens = 1_000_000, ssm = true,
            files = emptyList()   // preenchido quando publicarmos o modelo destilado
        )
    )
}
