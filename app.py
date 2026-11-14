"""Dashboard interativo da análise da Black Friday 2020 (base de nb_populacao_bf.ipynb).

Rodar com: streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Black Friday 2020 · GA4", layout="wide")
alt.data_transformers.disable_max_rows()

ALPHA = 0.05
COR_SERIE = "#2a78d6"
CORES_IC = {
    "Estimativa (amostra)": "#2a78d6",
    "Benchmark (base completa)": "#eb6834",
    "Valor real (população BF)": "#1baf7a",
}


# ----------------------------------------------------------------------------
# formatação
# ----------------------------------------------------------------------------
def br(valor, casas=2):
    """Número no formato brasileiro (1.234,56)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(valor, casas=1):
    return f"{br(valor * 100, casas)}%"


def p_valor_fmt(valor):
    return f"{valor:.2e}".replace(".", ",") if valor < 0.001 else br(valor, 4)


# ----------------------------------------------------------------------------
# dados e cálculos (mesma lógica do notebook)
# ----------------------------------------------------------------------------
def variaveis_testes(base):
    """As 4 variáveis usadas nos testes de hipótese (sem corte)."""
    com_carrinho = base[base["qt_carrinho"] > 0]
    return {
        "tx_sessoes_engajadas": base["fl_sessao_engajada"],
        "tx_sessoes_carrinho_conversao": (com_carrinho["qt_sucesso"] > 0).astype(int),
        "avg_tempo_sessao_seg": base["nu_tempo_sessao"].dropna(),
        "avg_transacoes": base["vr_transacao"].dropna(),
    }


def tabela_estatisticas(base):
    linhas = []
    for nome, valores in variaveis_testes(base).items():
        n = len(valores)
        dp = valores.std()
        linhas.append({
            "variável": nome,
            "n": n,
            "média": valores.mean(),
            "mediana": valores.median(),
            "desvio padrão": dp,
            "assimetria": valores.skew(),
            "erro padrão": dp / np.sqrt(n),
        })
    return pd.DataFrame(linhas).set_index("variável")


def teste_z_proporcao(nome, sucesso_base, sucesso_pop, sucesso_amostra):
    p0 = sucesso_base.mean()
    p_real = sucesso_pop.mean()
    k, n = int(sucesso_amostra.sum()), len(sucesso_amostra)
    p_hat = k / n
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_valor = 2 * stats.norm.sf(abs(z))
    z_critico = stats.norm.ppf(1 - ALPHA / 2)
    margem = z_critico * np.sqrt(p_hat * (1 - p_hat) / n)
    return {
        "nome": nome, "tipo": "proporção", "benchmark": p0, "estimativa": p_hat,
        "k": k, "n": n, "ic_low": p_hat - margem, "ic_high": p_hat + margem,
        "real": p_real, "z": z, "p_valor": p_valor,
        "n_p0": n * p0, "n_1_p0": n * (1 - p0),
    }


def teste_z_media(nome, valores_base, valores_pop, valores_amostra):
    mu0 = valores_base.mean()
    mu_real = valores_pop.mean()
    sigma = valores_pop.std(ddof=0)
    amostra = valores_amostra.dropna()
    n, media = len(amostra), amostra.mean()
    ep = sigma / np.sqrt(n)
    z = (media - mu0) / ep
    p_valor = 2 * stats.norm.sf(abs(z))
    z_critico = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "nome": nome, "tipo": "média", "benchmark": mu0, "estimativa": media,
        "n": n, "ic_low": media - z_critico * ep, "ic_high": media + z_critico * ep,
        "real": mu_real, "z": z, "p_valor": p_valor, "sigma": sigma, "ep": ep,
    }


@st.cache_resource(show_spinner="Carregando dados...")
def preparar():
    df = pd.read_csv("data/dados_ga4.csv")
    populacao_bf = df[df["fl_black_friday"] == 1]

    # amostra aleatória simples de usuários (95% de confiança, e = 2%, p = 0,5)
    z, e, p = 1.96, 0.02, 0.5
    usuarios_bf = pd.Series(populacao_bf["cd_pseudo"].unique())
    N = len(usuarios_bf)
    n0 = z**2 * p * (1 - p) / e**2
    n = int(np.ceil(n0 / (1 + (n0 - 1) / N)))
    usuarios_amostra = usuarios_bf.sample(n=n, random_state=42)
    df_amostra = populacao_bf[populacao_bf["cd_pseudo"].isin(usuarios_amostra)]

    def com_carrinho(base):
        return base[base["qt_carrinho"] > 0]["qt_sucesso"] > 0

    testes = [
        teste_z_proporcao(
            "Proporção 1: engajamento das sessões",
            df["fl_sessao_engajada"], populacao_bf["fl_sessao_engajada"],
            df_amostra["fl_sessao_engajada"],
        ),
        teste_z_proporcao(
            "Proporção 2: conversão das sessões com carrinho",
            com_carrinho(df), com_carrinho(populacao_bf), com_carrinho(df_amostra),
        ),
        teste_z_media(
            "Média 1: tempo de sessão (s)",
            df["nu_tempo_sessao"], populacao_bf["nu_tempo_sessao"],
            df_amostra["nu_tempo_sessao"],
        ),
        teste_z_media(
            "Média 2: ticket médio por transação",
            df["vr_transacao"], populacao_bf["vr_transacao"], df_amostra["vr_transacao"],
        ),
    ]

    return {
        "totais_base": (df["cd_pseudo"].nunique(), df["cd_sessao"].nunique()),
        "totais_pop": (populacao_bf["cd_pseudo"].nunique(), populacao_bf["cd_sessao"].nunique()),
        "totais_amostra": (df_amostra["cd_pseudo"].nunique(), df_amostra["cd_sessao"].nunique()),
        "populacao_bf": populacao_bf,
        "df_amostra": df_amostra,
        "calc_amostra": {"z": z, "e": e, "p": p, "N": N, "n0": n0, "n": n},
        "testes": testes,
    }


# ----------------------------------------------------------------------------
# gráficos
# ----------------------------------------------------------------------------
def corte_percentil(valores, percentil):
    if percentil >= 100:
        return valores
    return valores[valores <= valores.quantile(percentil / 100)]


def grafico_boxplot(valores, titulo):
    dados = pd.DataFrame({"valor": valores.to_numpy()})
    caixa = alt.Chart(dados).mark_boxplot(extent=1.5, size=40, color=COR_SERIE).encode(
        x=alt.X("valor:Q", title=None),
    )
    media = alt.Chart(dados).transform_aggregate(media="mean(valor)").mark_point(
        shape="triangle-up", size=120, filled=True, color="#eb6834",
    ).encode(
        x="media:Q",
        tooltip=[alt.Tooltip("media:Q", title="média", format=",.2f")],
    )
    return (caixa + media).properties(title=titulo, height=160, width="container")


def grafico_proporcao(flag, titulo):
    proporcao = (
        flag.map({True: "sim", False: "não"})
        .value_counts(normalize=True)
        .reindex(["sim", "não"])
        .fillna(0)
    )
    dados = pd.DataFrame({"categoria": proporcao.index, "proporção": proporcao.values})
    base = alt.Chart(dados).encode(
        x=alt.X("categoria:N", title=None, sort=["sim", "não"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("proporção:Q", scale=alt.Scale(domain=[0, 1.1]), axis=alt.Axis(format="%")),
    )
    barras = base.mark_bar(color=COR_SERIE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        tooltip=[alt.Tooltip("categoria:N"), alt.Tooltip("proporção:Q", format=".1%")],
    )
    rotulos = base.mark_text(dy=-8).encode(text=alt.Text("proporção:Q", format=".1%"))
    return (barras + rotulos).properties(title=titulo, height=160, width="container")


def grafico_histograma(valores, titulo, bins, log=False):
    """Histograma em densidade com curva KDE, como o sns.histplot(stat="density", kde=True).

    Com log=True, os bins e a KDE são calculados sobre log10(valor) (só valores > 0)
    e o eixo x fica em escala logarítmica; a densidade passa a ser por unidade de log10.
    """
    x = valores.to_numpy(dtype=float)
    if log:
        x = x[x > 0]
    # u = espaço onde os bins e a KDE são calculados (log10 do valor, ou o próprio valor)
    u = np.log10(x) if log else x
    voltar = (lambda v: 10 ** v) if log else (lambda v: v)

    densidades, bordas = np.histogram(u, bins=bins, density=True)
    contagens, _ = np.histogram(u, bins=bordas)
    folga = (bordas[1:] - bordas[:-1]) * 0.03  # espaço entre as colunas
    dados = pd.DataFrame({
        "início": voltar(bordas[:-1]), "fim": voltar(bordas[1:]),
        "x_ini": voltar(bordas[:-1] + folga), "x_fim": voltar(bordas[1:] - folga),
        "contagem": contagens, "densidade": densidades,
    })
    escala_x = alt.Scale(type="log") if log else alt.Scale()
    colunas = alt.Chart(dados).mark_bar(color=COR_SERIE, opacity=0.85).encode(
        x=alt.X("x_ini:Q", title=None, scale=escala_x),
        x2="x_fim:Q",
        y=alt.Y("densidade:Q", title="densidade (por log₁₀)" if log else "densidade"),
        tooltip=[
            alt.Tooltip("início:Q", title="de", format=",.2f"),
            alt.Tooltip("fim:Q", title="até", format=",.2f"),
            alt.Tooltip("contagem:Q", title="sessões", format=","),
            alt.Tooltip("densidade:Q", format=".4g"),
        ],
    )
    grafico = colunas
    if np.unique(u).size >= 2:
        grade = np.linspace(u.min(), u.max(), 200)
        curva = pd.DataFrame({"valor": voltar(grade), "densidade": stats.gaussian_kde(u)(grade)})
        linha = alt.Chart(curva).mark_line(color="#eb6834", strokeWidth=2).encode(
            x="valor:Q",
            y="densidade:Q",
            tooltip=[alt.Tooltip("valor:Q", format=",.2f"),
                     alt.Tooltip("densidade:Q", title="densidade (KDE)", format=".4g")],
        )
        grafico = colunas + linha
    return grafico.properties(title=titulo, height=300, width="container")


def render_analise(base, rotulo, key):
    st.subheader(f"Distribuição das features dos testes · {rotulo}")
    col_slider, col_bins = st.columns([3, 1])
    percentil = col_slider.slider(
        "Remover valores acima do percentil (variáveis contínuas)",
        min_value=90, max_value=100, value=99, key=f"percentil_{key}",
        help="100 = sem corte. Afeta só os boxplots e os histogramas.",
    )
    bins = col_bins.select_slider(
        "Bins dos histogramas", options=["auto", 10, 20, 30, 50, 80], value="auto", key=f"bins_{key}",
    )
    col_zeros, col_escala = st.columns([3, 1])
    sem_zeros = col_zeros.checkbox(
        "Remover sessões com tempo 0 no histograma", value=True, key=f"zeros_{key}",
        help="Como no notebook. O boxplot e a tabela continuam com os zeros, que entram no teste.",
    )
    escala = col_escala.radio(
        "Escala do eixo x dos histogramas", ["linear", "logarítmica"], horizontal=True,
        key=f"escala_{key}",
        help="Na logarítmica, os bins e a curva são calculados sobre log₁₀ do valor; valores 0 ficam de fora.",
    )
    log = escala == "logarítmica"
    sufixo = "" if percentil >= 100 else f" (≤ p{percentil})"

    tempo = corte_percentil(base["nu_tempo_sessao"].dropna(), percentil)
    ticket = corte_percentil(base["vr_transacao"].dropna(), percentil)
    com_carrinho = base[base["qt_carrinho"] > 0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.altair_chart(grafico_boxplot(tempo, f"avg_tempo_sessao_seg{sufixo}"))
    with c2:
        st.altair_chart(grafico_boxplot(ticket, f"avg_transacoes{sufixo}"))
    with c3:
        st.altair_chart(grafico_proporcao(base["fl_sessao_engajada"] == 1, "tx_sessoes_engajadas"))
    with c4:
        st.altair_chart(grafico_proporcao(com_carrinho["qt_sucesso"] > 0, "tx_sessoes_carrinho_conversao"))
    st.caption("Nos boxplots, o triângulo laranja marca a média e os bigodes vão até 1,5·IQR.")

    tempo_hist = base["nu_tempo_sessao"].dropna()
    filtro_zeros = ""
    if sem_zeros or log:
        tempo_hist = tempo_hist[tempo_hist > 0]
        filtro_zeros = " (> 0)"
    tempo_hist = corte_percentil(tempo_hist, percentil)
    sufixo_hist = f"{sufixo} · escala log" if log else sufixo

    h1, h2 = st.columns(2)
    with h1:
        st.altair_chart(grafico_histograma(
            tempo_hist, f"Histograma · tempo de sessão (s){filtro_zeros}{sufixo_hist}", bins, log,
        ))
    with h2:
        st.altair_chart(grafico_histograma(
            ticket, f"Histograma · valor da transação{sufixo_hist}", bins, log,
        ))

    st.subheader(f"Estatísticas das features dos testes · {rotulo}")
    tabela = tabela_estatisticas(base)
    st.dataframe(
        tabela.style.format(
            {"n": "{:,.0f}", "média": "{:,.4f}", "mediana": "{:,.4f}", "desvio padrão": "{:,.4f}",
             "assimetria": "{:,.4f}", "erro padrão": "{:,.4f}"},
            decimal=",", thousands=".",
        )
    )
    st.caption(
        "A tabela usa todos os dados, sem o corte de percentil. "
        "Nas variáveis binárias, a média é a proporção. Erro padrão = dp / √n."
    )


def grafico_ic(teste, formato):
    pontos = pd.DataFrame([
        {"tipo": "Estimativa (amostra)", "valor": teste["estimativa"]},
        {"tipo": "Benchmark (base completa)", "valor": teste["benchmark"]},
        {"tipo": "Valor real (população BF)", "valor": teste["real"]},
    ])
    ic = pd.DataFrame([{"low": teste["ic_low"], "high": teste["ic_high"]}])
    escala_x = alt.Scale(zero=False, nice=True)
    legenda = alt.Legend(
        orient="bottom", direction="horizontal", labelLimit=0, columns=3, symbolSize=120,
    )
    cor = alt.Color(
        "tipo:N", title=None,
        scale=alt.Scale(domain=list(CORES_IC), range=list(CORES_IC.values())),
        legend=legenda,
    )

    faixa = alt.Chart(ic).mark_rule(strokeWidth=3, color="#86b6ef").encode(
        x=alt.X("low:Q", title=None, scale=escala_x, axis=alt.Axis(format=formato)),
        x2="high:Q",
        tooltip=[alt.Tooltip("low:Q", title="IC inferior", format=formato),
                 alt.Tooltip("high:Q", title="IC superior", format=formato)],
    )
    benchmark = alt.Chart(pontos[pontos["tipo"] == "Benchmark (base completa)"]).mark_rule(
        strokeDash=[6, 4], strokeWidth=2, color=CORES_IC["Benchmark (base completa)"],
    ).encode(x="valor:Q")
    marcadores = alt.Chart(pontos).mark_point(filled=True, size=160, opacity=1).encode(
        x="valor:Q",
        color=cor,
        shape=alt.Shape("tipo:N", title=None, scale=alt.Scale(domain=list(CORES_IC)), legend=legenda),
        tooltip=[alt.Tooltip("tipo:N", title=" "), alt.Tooltip("valor:Q", format=formato)],
    )
    return (faixa + benchmark + marcadores).properties(height=140, width="container", title="IC de 95% da amostra")


# ----------------------------------------------------------------------------
# textos dos testes (do notebook)
# ----------------------------------------------------------------------------
DESCRICOES = [
    {
        "variavel": "`fl_sessao_engajada` (nominal dicotômica: engajada / não engajada)",
        "parametro": "p_BF, a proporção de sessões engajadas na Black Friday",
        "h0": "p_BF = p₀ (o engajamento na Black Friday é igual ao do site como um todo)",
        "h1": "p_BF ≠ p₀ (o engajamento na Black Friday é diferente)",
        "metodo": "teste Z para uma proporção e IC de 95% de Wald",
        "conclusao": (
            "Com 95% de confiança, o engajamento na Black Friday é **menor** que o do site, cerca de "
            "10 pontos percentuais a menos. Uma explicação provável é o tráfego de campanhas da Black "
            "Friday, com visitantes que entram, olham uma oferta e saem."
        ),
    },
    {
        "variavel": "sessão com carrinho (`qt_carrinho > 0`) que teve pelo menos uma compra (`qt_sucesso > 0`)",
        "parametro": "p_BF, a proporção de sessões com carrinho que converteram na Black Friday",
        "h0": "p_BF = p₀ (a conversão do carrinho na Black Friday é igual à do site como um todo)",
        "h1": "p_BF ≠ p₀ (a conversão do carrinho na Black Friday é diferente)",
        "metodo": "teste Z para uma proporção e IC de 95% de Wald",
        "conclusao": (
            "Não há evidência de que a conversão do carrinho mude na Black Friday. Com poucas sessões "
            "com carrinho na amostra, o IC ainda é largo o bastante para conter a taxa do site. A decisão "
            "coincide com a realidade, já que a população e a base completa têm praticamente a mesma taxa."
        ),
    },
    {
        "variavel": "`nu_tempo_sessao` (contínua, em segundos, incluindo as sessões com tempo 0)",
        "parametro": "μ_BF, o tempo médio de sessão na Black Friday",
        "h0": "μ_BF = μ₀ (o tempo médio de sessão na Black Friday é igual ao do site como um todo)",
        "h1": "μ_BF ≠ μ₀ (o tempo médio de sessão na Black Friday é diferente)",
        "metodo": "teste Z para uma média, com σ conhecido (calculado sobre a população da Black Friday)",
        "conclusao": (
            "Com 95% de confiança, as sessões da Black Friday são **mais longas** que as do site. Junto com "
            "o teste 1, o perfil é: menos sessões engajadas, mas mais longas entre as que ficam. O tempo de "
            "sessão é muito assimétrico, mas com n grande o Teorema Central do Limite sustenta o teste Z."
        ),
    },
    {
        "variavel": "`vr_transacao` (contínua, só nas sessões com compra)",
        "parametro": "μ_BF, o ticket médio por transação na Black Friday",
        "h0": "μ_BF = μ₀ (o ticket médio na Black Friday é igual ao do site como um todo)",
        "h1": "μ_BF ≠ μ₀ (o ticket médio na Black Friday é diferente)",
        "metodo": "teste Z para uma média, com σ conhecido (calculado sobre a população da Black Friday)",
        "conclusao": (
            "Com 95% de confiança, o ticket médio na Black Friday é **maior** que o do site, e a decisão "
            "coincide com a realidade. A estimativa pontual fica bem acima do valor real, efeito das poucas "
            "compras na amostra. **Ressalva:** com poucas transações e valores assimétricos, a normalidade "
            "da média amostral é apenas aproximada."
        ),
    },
]


def fmt_valor(teste, valor):
    return pct(valor) if teste["tipo"] == "proporção" else br(valor)


def decisao(teste):
    return "rejeita H₀" if teste["p_valor"] < ALPHA else "não rejeita H₀"


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
dados = preparar()
populacao_bf = dados["populacao_bf"]
df_amostra = dados["df_amostra"]
calc = dados["calc_amostra"]
testes = dados["testes"]

st.title("Black Friday 2020 · Google Merchandise Store")
st.info(
    "**Hipótese do estudo:** a navegação na semana da Black Friday difere do restante do ano.",
    icon="🔎",
)

aba_pop, aba_amostra, aba_testes = st.tabs(["População", "Amostra", "Testes de hipótese"])

with aba_pop:
    st.markdown(
        """
Os dados vêm da **amostra pública do Google Analytics 4** que o Google disponibiliza no **BigQuery**
(`ga4_obfuscated_sample_ecommerce`). São os eventos de e-commerce da
[Google Merchandise Store](https://shop.merch.google/) entre **novembro de 2020 e janeiro de 2021**,
agregados aqui por sessão.
"""
    )
    usuarios, sessoes = dados["totais_base"]
    m1, m2 = st.columns(2)
    m1.metric("Usuários na base completa", br(usuarios, 0))
    m2.metric("Sessões na base completa", br(sessoes, 0))

    st.divider()
    st.header("População: Black Friday 2020")
    st.markdown(
        "A população são as **sessões dos usuários que acessaram o site na Black Friday de 2020** "
        "(`fl_black_friday = 1`). É dela que a amostra é sorteada."
    )
    usuarios, sessoes = dados["totais_pop"]
    m1, m2 = st.columns(2)
    m1.metric("Usuários na Black Friday", br(usuarios, 0))
    m2.metric("Sessões na Black Friday", br(sessoes, 0))

    render_analise(populacao_bf, "População BF", "pop")

with aba_amostra:
    st.header("Amostra da Black Friday 2020")
    st.markdown(
        f"""
A amostra foi gerada por **amostragem aleatória simples de usuários** da população, com
**95% de grau de confiança** (z = 1,96), **2% de margem de erro**, **p = 0,50** (o caso mais
conservador) e **semente = 42**. Entram na amostra todas as sessões da Black Friday dos usuários sorteados.

O tamanho parte da fórmula para proporções e depois recebe a correção para população finita,
já que a população tem N = {br(calc['N'], 0)} usuários:
"""
    )
    f1, f2 = st.columns(2)
    with f1:
        st.latex(r"n_0 = \frac{z^2\,p\,(1-p)}{e^2} = \frac{1{,}96^2 \cdot 0{,}5 \cdot 0{,}5}{0{,}02^2} = "
                 + br(calc["n0"], 0))
    with f2:
        st.latex(r"n = \left\lceil \frac{n_0}{1 + \frac{n_0 - 1}{N}} \right\rceil = "
                 r"\left\lceil \frac{" + br(calc["n0"], 0) + r"}{1 + \frac{" + br(calc["n0"] - 1, 0)
                 + r"}{" + br(calc["N"], 0) + r"}} \right\rceil = " + br(calc["n"], 0))

    usuarios, sessoes = dados["totais_amostra"]
    m1, m2 = st.columns(2)
    m1.metric("Usuários na amostra", br(usuarios, 0))
    m2.metric("Sessões na amostra", br(sessoes, 0))

    render_analise(df_amostra, "Amostra BF", "amo")

with aba_testes:
    st.header("Testes de hipótese")
    st.markdown(
        "Os quatro testes comparam a **amostra da Black Friday** com um **benchmark calculado sobre a "
        f"base completa** (todo o site). Todos usam o teste Z, bilateral, com α = {br(ALPHA, 2)}."
    )

    st.subheader("Benchmarks (base completa)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("p₀ · sessões engajadas", pct(testes[0]["benchmark"]))
    b2.metric("p₀ · conversão do carrinho", pct(testes[1]["benchmark"]))
    b3.metric("μ₀ · tempo de sessão", f"{br(testes[2]['benchmark'], 1)} s")
    b4.metric("μ₀ · ticket médio", br(testes[3]["benchmark"]))

    st.subheader("Resumo dos testes")
    resumo = pd.DataFrame([
        {
            "teste": t["nome"],
            "benchmark": fmt_valor(t, t["benchmark"]),
            "estimativa (amostra)": fmt_valor(t, t["estimativa"]),
            "n": br(t["n"], 0),
            "IC 95%": f"[{fmt_valor(t, t['ic_low'])} ; {fmt_valor(t, t['ic_high'])}]",
            "valor real (população)": fmt_valor(t, t["real"]),
            "Z": br(t["z"], 2),
            "p-valor": p_valor_fmt(t["p_valor"]),
            "decisão": decisao(t),
        }
        for t in testes
    ]).set_index("teste")
    if st.toggle("Mostrar tabela de resumo", value=True, key="mostrar_resumo"):
        st.dataframe(resumo)

    for i, (teste, desc) in enumerate(zip(testes, DESCRICOES), start=1):
        st.divider()
        st.subheader(f"Teste {i} · {teste['nome']}")
        simbolo = "p₀" if teste["tipo"] == "proporção" else "μ₀"
        col_texto, col_tabela = st.columns([3, 2])
        with col_texto:
            st.markdown(
                f"""
- **Variável:** {desc['variavel']}
- **Parâmetro:** {desc['parametro']}
- **Benchmark:** {simbolo} = **{fmt_valor(teste, teste['benchmark'])}** (base completa)
- **H₀:** {desc['h0']}
- **H₁:** {desc['h1']}
- **Nível de significância:** α = {br(ALPHA, 2)} (bilateral)
- **Método:** {desc['metodo']}
"""
            )
        with col_tabela:
            linhas = [("nível de significância", f"α = {br(ALPHA, 2)}")]
            if teste["tipo"] == "proporção":
                linhas += [
                    ("p̂ (amostra)", f"{teste['k']}/{teste['n']} = {pct(teste['estimativa'], 2)}"),
                    ("n·p₀ | n·(1 − p₀)", f"{br(teste['n_p0'], 1)} | {br(teste['n_1_p0'], 1)}"),
                ]
            else:
                linhas += [
                    ("x̄ (amostra)", f"{br(teste['estimativa'])} (n = {br(teste['n'], 0)})"),
                    ("σ (população BF)", br(teste["sigma"])),
                ]
            dentro = teste["ic_low"] <= teste["real"] <= teste["ic_high"]
            linhas += [
                ("IC 95%", f"[{fmt_valor(teste, teste['ic_low'])} ; {fmt_valor(teste, teste['ic_high'])}]"),
                ("valor real (população)", f"{fmt_valor(teste, teste['real'])} ({'dentro' if dentro else 'fora'} do IC)"),
                ("estatística de teste", f"Z = {br(teste['z'], 2)}"),
                ("p-valor", p_valor_fmt(teste["p_valor"])),
                ("decisão", decisao(teste)),
            ]
            st.table(pd.DataFrame(linhas, columns=["", "valor"]).set_index(""))

        formato = ".1%" if teste["tipo"] == "proporção" else ",.2f"
        st.altair_chart(grafico_ic(teste, formato))
        st.markdown(f"**Conclusão:** {desc['conclusao']}")
