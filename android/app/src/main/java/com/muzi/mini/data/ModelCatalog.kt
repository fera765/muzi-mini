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
            id = "qwen2.5-0.5b-int4",
            label = "Qwen2.5 0.5B Instruct (INT4) — leve e rapido",
            paramsB = "0.5B", approxRamMB = 450, contextTokens = 32768, ssm = false,
            files = listOf(
                // Ajustar para o repo ONNX GenAI exato que voce usar.
                ModelFile("genai_config.json", hf("onnx-community/Qwen2.5-0.5B-Instruct", "onnx/genai_config.json"), 1),
                ModelFile("model.onnx", hf("onnx-community/Qwen2.5-0.5B-Instruct", "onnx/model.onnx"), 1),
                ModelFile("model.onnx.data", hf("onnx-community/Qwen2.5-0.5B-Instruct", "onnx/model.onnx.data"), 500),
                ModelFile("tokenizer.json", hf("onnx-community/Qwen2.5-0.5B-Instruct", "tokenizer.json"), 3)
            )
        ),
        ModelSpec(
            id = "qwen2.5-1.5b-int4",
            label = "Qwen2.5 1.5B Instruct (INT4) — qualidade alta (professor)",
            paramsB = "1.5B", approxRamMB = 1300, contextTokens = 32768, ssm = false,
            files = listOf(
                ModelFile("genai_config.json", hf("onnx-community/Qwen2.5-1.5B-Instruct", "onnx/genai_config.json"), 1),
                ModelFile("model.onnx", hf("onnx-community/Qwen2.5-1.5B-Instruct", "onnx/model.onnx"), 1),
                ModelFile("model.onnx.data", hf("onnx-community/Qwen2.5-1.5B-Instruct", "onnx/model.onnx.data"), 1300),
                ModelFile("tokenizer.json", hf("onnx-community/Qwen2.5-1.5B-Instruct", "tokenizer.json"), 3)
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
