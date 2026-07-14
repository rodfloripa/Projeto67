<div align="justify">

# 1. Mini Transformer com Mixture of Experts (MoE)

Este projeto implementa um <b>Mini Transformer autoregressivo</b> utilizando <b>PyTorch</b> e introduz uma das arquiteturas mais importantes dos Large Language Models modernos: <b>Mixture of Experts (MoE)</b>.

O objetivo principal é demonstrar como transformar um Transformer denso tradicional em um <b>Sparse Transformer</b>, permitindo aumentar significativamente a capacidade do modelo sem aumentar proporcionalmente o custo computacional.

A arquitetura implementada neste projeto segue a mesma ideia utilizada em modelos modernos como:

* Mistral AI — Mixtral
* Google DeepMind — Switch Transformer
* DeepSeek — DeepSeekMoE
* xAI — Grok MoE
* Databricks — DBRX

</div>

---

# 2. Arquitetura Geral

<div align="justify">

O modelo implementado possui a seguinte estrutura:

```text id="0l2zrf"
Embedding
    ↓
Positional Encoding
    ↓
Transformer Block
    ├── Multi-Head Self-Attention
    ├── LayerNorm
    ├── MoE Layer
    │      ├── Router
    │      ├── Expert 1
    │      ├── Expert 2
    │      ├── Expert 3
    │      └── Expert N
    ↓
Linear Head
    ↓
Predição do próximo token
```

A principal modificação em relação a um Transformer tradicional ocorre no bloco Feed Forward Network (FFN), que foi substituído por múltiplos experts especializados controlados dinamicamente por um Router.

</div>

---

# 3. Dataset Sintético

<div align="justify">

O projeto utiliza um pequeno corpus textual sintético relacionado a inteligência artificial, Transformers e Sparse Models.

Exemplo:

```python id="pjhx9q"
sentences = [
    "transformers use attention",
    "mixture of experts is sparse",
    "language models predict tokens",
    "routing tokens to experts"
]
```

O objetivo é permitir que o modelo aprenda padrões sequenciais básicos enquanto possibilita visualizar o comportamento dos experts.

O texto é tokenizado em nível de caracteres.

</div>

---

# 4. Positional Encoding

<div align="justify">

Transformers não possuem recorrência. Portanto, é necessário inserir explicitamente informações de posição dos tokens.

O projeto utiliza Positional Encoding senoidal:

PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)

PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)

Trecho principal da implementação:

```python id="8xq0pn"
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

Esse mecanismo permite ao modelo entender relações relativas entre posições sem utilizar RNNs ou LSTMs.

</div>

---

# 5. Multi-Head Self-Attention

<div align="justify">

O mecanismo de Self-Attention permite que cada token observe todos os demais tokens da sequência.

A fórmula principal da atenção é:

Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V

O bloco principal da implementação é:

```python id="e15t69"
qkv = self.qkv(x)

q, k, v = qkv.chunk(3, dim=-1)

attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

attn = F.softmax(attn, dim=-1)

out = attn @ v
```

Explicação:

* `q` representa Queries
* `k` representa Keys
* `v` representa Values

A multiplicação:

```python id="g8gcb1"
q @ k.transpose(-2, -1)
```

mede similaridade entre tokens.

O softmax transforma essas similaridades em pesos probabilísticos de atenção.

O modelo utiliza múltiplas cabeças de atenção paralelas, permitindo aprender diferentes relações semânticas simultaneamente.

</div>

---

# 6. Expert Networks

<div align="justify">

Cada expert é uma pequena rede neural independente responsável por processar subconjuntos específicos de tokens.

Implementação principal:

```python id="0h6m2m"
class Expert(nn.Module):

    def __init__(self, d_model, d_ff):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
```

Explicação:

* primeira camada linear expande a dimensionalidade
* GELU adiciona não linearidade
* segunda camada projeta novamente para o espaço original

Cada expert aprende representações diferentes durante o treinamento.

</div>

---

# 7. Router

<div align="justify">

O Router é o componente central do MoE.

Ele decide qual expert será utilizado para cada token.

Implementação principal:

```python id="gpx2w6"
class Router(nn.Module):

    def __init__(self, d_model, num_experts):

        super().__init__()

        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):

        logits = self.gate(x)

        probs = F.softmax(logits, dim=-1)

        return probs
```

A projeção linear gera scores para cada expert.

O softmax converte esses scores em probabilidades:

P(expert|token)=softmax(Wx+b)

Cada token então é roteado para o expert com maior probabilidade.

</div>

---

# 8. Top-K Routing

<div align="justify">

O projeto utiliza Top-1 Routing.

Ou seja:

```text id="4n9sdy"
Cada token é enviado apenas ao expert mais provável
```

Trecho principal:

```python id="1dys58"
topk_probs, topk_indices = torch.topk(
    routing_probs,
    self.top_k,
    dim=-1
)
```

O `torch.topk` seleciona os experts mais relevantes para cada token.

Essa estratégia reduz drasticamente o custo computacional porque apenas alguns experts são ativados.

</div>

---

# 9. MoE Layer

<div align="justify">

A MoE Layer substitui diretamente o Feed Forward Network tradicional do Transformer.

Implementação principal:

```python id="njlwmk"
for i, expert in enumerate(self.experts):

    mask = (expert_idx == i)

    if mask.any():

        selected = x[mask]

        expert_output = expert(selected)

        output[mask] += expert_output * expert_prob[mask]
```

Explicação detalhada:

* o Router escolhe o expert
* uma máscara identifica os tokens daquele expert
* apenas esses tokens são processados
* a saída é ponderada pela probabilidade do Router

Isso cria computação esparsa dinâmica.

Cada token ativa apenas uma pequena parte da rede.

</div>

---

# 10. Transformer Block

<div align="justify">

Cada bloco Transformer contém:

```text id="uxckhx"
LayerNorm
    ↓
Self-Attention
    ↓
Residual
    ↓
LayerNorm
    ↓
MoE Layer
    ↓
Residual
```

Trecho principal:

```python id="yhlgph"
x = x + self.attn(self.ln1(x))

x = x + self.moe(self.ln2(x))
```

As residual connections ajudam na estabilidade do treinamento e evitam degradação de gradientes.

</div>

---

# 11. Forward Pass do Modelo

<div align="justify">

O forward completo do modelo segue o fluxo:

```python id="pdknwo"
x = self.embedding(idx)

x = self.pos_encoding(x)

for block in self.blocks:
    x = block(x)

x = self.ln_f(x)

logits = self.head(x)
```

Explicação:

* embeddings convertem tokens em vetores densos
* positional encoding adiciona informação temporal
* os blocos Transformer processam dependências contextuais
* a camada final gera probabilidades sobre o vocabulário

</div>

---

# 12. Função de Loss

<div align="justify">

O treinamento utiliza Cross Entropy Loss:

L=-\sum y\log(\hat{y})

Implementação:

```python id="37oyij"
loss = F.cross_entropy(
    logits_flat,
    targets_flat
)
```

O modelo aprende a prever o próximo token da sequência.

</div>

---

# 13. Loop de Treinamento

<div align="justify">

Trecho principal:

```python id="y2ws56"
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Explicação:

* `zero_grad()` limpa gradientes anteriores
* `backward()` calcula backpropagation
* `step()` atualiza os pesos

Durante o treinamento o modelo aprende:

* relações sintáticas
* dependências sequenciais
* padrões estatísticos
* roteamento dos experts

</div>

---

# 14. Geração Autoregressiva

<div align="justify">

A geração de texto ocorre autoregressivamente.

Trecho principal:

```python id="3wl7e9"
logits = logits[:, -1, :]

probs = F.softmax(logits, dim=-1)

next_token = torch.multinomial(probs, 1)
```

Fluxo:

```text id="pr0svl"
Token atual
    ↓
Predição do próximo token
    ↓
Novo token inserido na sequência
    ↓
Repetição do processo
```

O modelo conseguiu gerar frases semanticamente coerentes:

```text id="49pt3e"
mixture of experts is sparse
language models predict tokens
routing tokens to experts
```

</div>

---

# 15. Visualização dos Experts

![download](https://github.com/rodfloripa/Projeto67/blob/main/download.png?raw=true)

![download](https://github.com/rodfloripa/Projeto67/blob/main/download (1).png?raw=true)

<div align="justify">

O projeto registra a utilização de cada expert ao longo do treinamento.

Trecho principal:

```python id="u5a0v5"
expert_usage[i] += mask.sum()
```

Isso permite visualizar:

* balanceamento dos experts
* frequência de ativação
* especialização emergente
* comportamento do Router

Com mais treinamento, diferentes experts começam a especializar-se em diferentes padrões linguísticos.

</div>

---

# 16. Load Balancing Loss

<div align="justify">

Uma melhoria importante seria implementar balanceamento de carga entre experts.

A loss auxiliar utilizada no Switch Transformer é:

L_{aux}=N\sum_i f_i P_i

Essa loss evita que apenas alguns experts dominem completamente o roteamento.

Sem balanceamento podem surgir:

* experts mortos
* colapso de roteamento
* especialização degenerada

</div>

---

# 17. Resultados

<div align="justify">

O modelo conseguiu aprender padrões linguísticos coerentes mesmo utilizando um dataset extremamente pequeno.

Os principais conceitos implementados foram:

| Conceito                | Implementado |
| ----------------------- | ------------ |
| Sparse Activation       | Sim          |
| Conditional Computation | Sim          |
| Token Routing           | Sim          |
| Expert Networks         | Sim          |
| Sparse FFN              | Sim          |
| Dynamic Compute         | Sim          |

O projeto demonstra claramente os princípios fundamentais utilizados em Sparse Large Language Models modernos.

</div>

---

# 18. Conclusão

<div align="justify">

Este projeto implementou um Mini Transformer com Mixture of Experts reproduzindo os principais mecanismos utilizados em arquiteturas Sparse modernas.

A substituição do Feed Forward Network tradicional por experts especializados permitiu introduzir:

* computação condicional
* roteamento dinâmico
* ativação esparsa
* especialização emergente

Mesmo em pequena escala, o modelo demonstrou funcionamento estável e geração coerente de texto, mostrando na prática como arquiteturas MoE conseguem aumentar drasticamente capacidade sem aumentar proporcionalmente o custo computacional.

Além disso, o projeto fornece uma excelente base para futuras extensões, incluindo:

* Noisy Routing
* Top-K Experts
* Shared Experts
* Capacity Factor
* Expert Parallelism
* Distributed MoE

O resultado final aproxima a implementação das arquiteturas utilizadas em modelos modernos como Mixtral, Switch Transformer e DeepSeekMoE, permitindo compreender profundamente os fundamentos dos Sparse Large Language Models.

</div>
