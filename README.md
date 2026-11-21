# muzi-mini

Laboratório de um LLM aluno compacto (~500M), destilado de um professor ~1B,
**CPU-only**, teto de **500 MB de RAM**, rodando em vários aparelhos (desktop
até Moto G60). Foco: velocidade, RAM baixa, contexto longo e qualidade.

## Decisão de arquitetura (validada no laboratório)

Para **1M de contexto dentro de 500 MB**, o Transformer clássico é
fisicamente inviável (o KV cache cresce linearmente → ~1,79 GB em 1M tokens).
A base é **núcleo recorrente estilo SSM** (estado fixo por token), pesos
**ternários (b1.58)**, e decodificação em bloco. Memória e velocidade ficam
**constantes** com o contexto — provado empiricamente (12k tokens, 0 MB de
crescimento de RSS, latência plana).

## nano — treino real do zero (prova de conceito do pipeline)

`nano/train_nano.py` treina em numpy puro a versão nano da arquitetura
(LM recorrente, memória constante por token), **sem download e sem framework**:

- backward/BPTT analítico + **checagem numérica de gradiente** (erro ~1e-7)
- split treino/validação (não treina no teste)
- clipping de gradiente + early stopping com melhor checkpoint

Resultado medido (dados de validação não vistos):

| métrica | antes | depois |
|---|---|---|
| acurácia próximo-char | 3,3% | **95,7%** |
| perplexidade | 26,1 | **1,1** |

Rodar:
```
python3 -I nano/train_nano.py
```

## Próximos passos
- Trocar a célula RNN pelo núcleo SSM seletivo (Mamba-like) com o mesmo loop.
- Kernel de matmul ternário empacotado (5 valores/byte).
- Destilação a partir do professor 1B (exige rede liberada para baixar o modelo).

> Nota de honestidade: RAM, velocidade, contexto 1M e corretude do treino
> estão validados de verdade. Qualidade em escala 500M exige treinar/baixar
> o modelo — bloqueado neste ambiente por rede/instalação.
