"""
Harness de avaliacao de QUALIDADE — aluno vs professor, 10 cenarios +
function calling. Pluggavel: aceita qualquer modelo que implemente
generate(prompt)->str e, p/ tools, generate_tool(prompt, tools)->dict.

HONESTIDADE (GR-9): este arquivo mede qualidade REAL quando um modelo e
plugado. O DummyModel abaixo serve so para provar que o encanamento roda —
seus numeros NAO sao qualidade do aluno, sao ruido de placeholder.
A meta-guardrail: aluno >= 95% da pontuacao do professor em cada cenario.
"""
import json, re, random
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------
# Protocolo de modelo
# --------------------------------------------------------------------------
class ModelInterface:
    def generate(self, prompt: str) -> str: raise NotImplementedError
    def generate_tool(self, prompt: str, tools: list) -> dict: raise NotImplementedError

# --------------------------------------------------------------------------
# Cenarios: (nome, prompt, funcao_score(resposta)->0..1)
# datasets de teste SEPARADOS do treino (GR: nao treinar no teste)
# --------------------------------------------------------------------------
def score_math(ans):      return 1.0 if re.search(r"\b42\b", ans) else 0.0
def score_logic(ans):     return 1.0 if re.search(r"\bfalso|false\b", ans.lower()) else 0.0
def score_code(ans):      return 1.0 if "def " in ans and "return" in ans else 0.0
def score_pt(ans):        return 1.0 if "porque" in ans.lower() else 0.0
def score_contains(k):    return lambda ans: 1.0 if k.lower() in ans.lower() else 0.0

SCENARIOS = [
    ("raciocinio_logico", "Se todo A e B, e nenhum B e C, algum A e C? Responda falso/verdadeiro.", score_logic),
    ("matematica",        "Quanto e 6 x 7? Responda so o numero.", score_math),
    ("programacao",       "Escreva em Python uma funcao que soma dois numeros.", score_code),
    ("portugues",         "Explique por que o ceu e azul em uma frase comecando com 'O ceu e azul porque'.", score_pt),
    ("conhecimento_geral","Qual a capital da Franca?", score_contains("paris")),
    ("seguir_instrucoes", "Responda apenas com a palavra OK.", score_contains("ok")),
    ("retencao_contexto", "Meu nome e Mateus. Qual e o meu nome?", score_contains("mateus")),
    ("resumos",           "Resuma: O gato subiu no telhado e dormiu.", score_contains("gato")),
    ("criatividade",      "Invente um nome para uma cafeteria.", lambda a: 1.0 if len(a.strip())>2 else 0.0),
    ("conversacao",       "Oi, tudo bem?", lambda a: 1.0 if len(a.strip())>2 else 0.0),
]

# --------------------------------------------------------------------------
# Function calling: valida JSON estruturado e selecao de ferramenta correta
# --------------------------------------------------------------------------
TOOLS = [{"name": "get_weather", "params": {"city": "string"}},
         {"name": "add",         "params": {"a": "number", "b": "number"}}]

def score_tool_call(result):
    """result esperado: {'name':..., 'arguments': {...}}"""
    if not isinstance(result, dict): return 0.0
    if result.get("name") != "get_weather": return 0.0
    args = result.get("arguments", {})
    return 1.0 if isinstance(args, dict) and "city" in args else 0.0

FUNCTION_CALL_PROMPT = "Qual a temperatura em Sao Paulo? Use as ferramentas."

# --------------------------------------------------------------------------
# Runner + guardrail 95%
# --------------------------------------------------------------------------
def evaluate_model(model: ModelInterface):
    scores = {}
    for name, prompt, scorer in SCENARIOS:
        try:
            scores[name] = scorer(model.generate(prompt))
        except NotImplementedError:
            scores[name] = None
    try:
        scores["function_calling"] = score_tool_call(model.generate_tool(FUNCTION_CALL_PROMPT, TOOLS))
    except NotImplementedError:
        scores["function_calling"] = None
    return scores

def regression_gate(student_scores, teacher_scores, threshold=0.95):
    """Retorna (passou, detalhes). Aluno deve atingir >=95% do professor por cenario."""
    details = {}
    passed = True
    for k, ts in teacher_scores.items():
        ss = student_scores.get(k)
        if ts is None or ss is None:
            details[k] = ("PENDENTE", ss, ts); continue
        ratio = 1.0 if ts == 0 else ss / ts
        ok = ratio >= threshold
        passed = passed and ok
        details[k] = ("OK" if ok else "FALHA", round(ratio, 3))
    return passed, details

# --------------------------------------------------------------------------
# DummyModel: SO prova o encanamento. NAO e qualidade do aluno. (GR-9)
# --------------------------------------------------------------------------
class DummyModel(ModelInterface):
    def __init__(self, seed): random.seed(seed)
    def generate(self, prompt):
        canned = {"6 x 7": "42", "falso": "falso", "Python": "def soma(a,b):\n return a+b",
                  "ceu": "O ceu e azul porque a luz se espalha", "Franca": "Paris",
                  "OK": "OK", "Mateus": "Mateus", "gato": "O gato dormiu"}
        for k, v in canned.items():
            if k.lower() in prompt.lower(): return v
        return "resposta generica"
    def generate_tool(self, prompt, tools):
        return {"name": "get_weather", "arguments": {"city": "Sao Paulo"}}

if __name__ == "__main__":
    print("HARNESS DE QUALIDADE — encanamento (DummyModel, NAO e qualidade real)\n")
    teacher = evaluate_model(DummyModel(1))
    student = evaluate_model(DummyModel(2))
    passed, details = regression_gate(student, teacher)
    for k, v in details.items():
        print(f"  {k:<20} {v}")
    print(f"\nPortao 95% (placeholder): {'PASSOU' if passed else 'FALHOU'}")
    print("\n>>> Numeros acima sao do DummyModel so para validar o encanamento.")
    print(">>> Qualidade REAL exige o aluno ternario treinado plugado aqui.")
