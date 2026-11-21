# Muzi Mini — app Android (ONNX Runtime GenAI, CPU-only)

App para Moto G60 (e qualquer Android 8+) que baixa modelos ONNX e roda
inferência **CPU-only** com ONNX Runtime GenAI (NEON). Catálogo de modelos
com download no próprio aparelho, chat, e auto-config de threads/RAM.

## O que já está no projeto
- **Catálogo de modelos** (`data/ModelCatalog.kt`): Qwen2.5 0.5B/1.5B INT4 e o
  slot do nosso `muzi-500m-ssm` (1M de contexto real quando publicado).
- **Download no aparelho** (`data/ModelDownloader.kt`): baixa do HuggingFace
  direto no celular (a internet do aparelho não tem o bloqueio do sandbox).
- **Motor** (`engine/OnnxLlmEngine.kt`): ONNX Runtime GenAI, geração em streaming.
- **Auto-config** (`engine/HardwareConfig.kt`): threads pelos núcleos, guardrail
  de 500 MB de RAM, checagem se o modelo cabe antes de carregar.
- **UI** (`MainActivity.kt`): abas Modelos e Chat em Jetpack Compose.

## Como compilar
Precisa do Android Studio (SDK + NDK) — **não dá para compilar o .apk no
sandbox do Claude**; este é o código-fonte para você buildar.
```
cd android
./gradlew assembleRelease     # ou abrir no Android Studio e Run
```
O APK sai em `app/build/outputs/apk/`.

## Limites honestos (leia antes de testar)
1. **1M de contexto real só com modelo SSM.** Os modelos Transformer do
   catálogo (Qwen) têm contexto limitado pela RAM — o KV cache cresce. O
   `muzi-500m-ssm` é o que destrava 1M de verdade, e ainda está em treino.
2. **URLs do catálogo são exemplos** — ajuste `ModelCatalog.kt` para o repo
   ONNX GenAI exato (pasta com `model.onnx` + `.data` + `genai_config.json` +
   `tokenizer.json`). Nem todo repo Qwen ONNX segue esse layout.
3. **Versão do onnxruntime-genai**: fixei 0.4.0; a API muda entre releases.
   Se trocar a versão, ajuste `OnnxLlmEngine.kt`.
4. "Velocidade absurda" depende do modelo e da quantização; no Moto G60,
   espere dezenas de tok/s em modelos pequenos INT4 (ver `reports/`).

## Próximo passo
Publicar o `muzi-500m-ssm` (destilado + ternário) e preencher seus arquivos
no catálogo — aí o app roda o nosso runtime com 1M de contexto.
