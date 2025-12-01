# app.py
    import streamlit as st
    from math import exp, factorial
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Poisson Bets — Mini App", layout="centered")

    st.title("Mini Algoritmo Poisson — App")
    st.markdown("Introduce los promedios de goles (últimos 5–10 partidos) y la app calculará probabilidades.")

    # Inputs
    st.header("Datos de entrada")
    col1, col2 = st.columns(2)
    with col1:
        A_local = st.number_input("Aₗ = Promedio goles anotados (local)", value=1.8, format="%.2f")
        R_local = st.number_input("Rₗ = Promedio goles recibidos (local)", value=1.2, format="%.2f")
    with col2:
        A_vis = st.number_input("Aᵥ = Promedio goles anotados (visitante)", value=1.3, format="%.2f")
        R_vis = st.number_input("Rᵥ = Promedio goles recibidos (visitante)", value=1.6, format="%.2f")

    max_goles = st.slider("Calcular probabilidades hasta k goles (ambos equipos)", 3, 8, 6)

    # Lambda calculations
    lambda_local = A_local * R_vis
    lambda_vis = A_vis * R_local

    st.markdown(f"**λ Local = {lambda_local:.3f}**  
**λ Visitante = {lambda_vis:.3f}**")

    # Poisson function
    def poisson_p(lambda_, k):
        return exp(-lambda_) * (lambda_ ** k) / factorial(k)

    # build probability arrays
    p_local = [poisson_p(lambda_local, k) for k in range(max_goles+1)]
    p_vis = [poisson_p(lambda_vis, k) for k in range(max_goles+1)]

    # Normalize (small rounding issues)
    p_local = np.array(p_local)
    p_vis = np.array(p_vis)
    p_local /= p_local.sum()
    p_vis /= p_vis.sum()

    # Score probability matrix
    matrix = np.outer(p_local, p_vis)
    index = [f"{k}" for k in range(max_goles+1)]
    columns = [f"{k}" for k in range(max_goles+1)]
    df_matrix = pd.DataFrame(matrix, index=index, columns=columns)

    st.header("Matriz de probabilidades de marcador (Local x Visitante)")
    st.dataframe(df_matrix.style.format("{:.3%}"))

    # Most probable score
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_prob_score = f"{max_idx[0]} - {max_idx[1]}"
    st.markdown(f"**Marcador más probable:** {most_prob_score}  ({matrix[max_idx]*100:.2f}%)")

    # Over / BTTS calculations
    totals = {}
    for total in range(0, (max_goles*2)+1):
        s = 0.0
        for i in range(max_goles+1):
            j = total - i
            if 0 <= j <= max_goles:
                s += matrix[i, j]
        totals[total] = s

    def prob_over(n):
        return sum(v for k,v in totals.items() if k > n)

    prob_over_0_5 = prob_over(0)
    prob_over_1_5 = prob_over(1)
    prob_over_2_5 = prob_over(2)
    prob_over_3_5 = prob_over(3)

    prob_btts = 1 - (matrix[:,0].sum() + matrix[0,:].sum() - matrix[0,0])

    # Gol en primer tiempo (aprox)
    lambda_total = lambda_local + lambda_vis
    prob_1T_any_goal_approx = 1 - poisson_p(0.48 * lambda_total, 0)

    st.header("Mercados rápidos")
    st.write(f"Over 0.5: **{prob_over_0_5:.2%}**")
    st.write(f"Over 1.5: **{prob_over_1_5:.2%}**")
    st.write(f"Over 2.5: **{prob_over_2_5:.2%}**")
    st.write(f"Over 3.5: **{prob_over_3_5:.2%}**")
    st.write(f"BTTS (Ambos marcan): **{prob_btts:.2%}**")
    st.write(f"Prob. de al menos 1 gol en 1er tiempo (estimada): **{prob_1T_any_goal_approx:.2%}**")

    # Plots
    st.header("Visualizaciones")
    fig, ax = plt.subplots()
    ax.bar(range(max_goles+1), p_local)
    ax.set_title("Distribución goles — Local")
    ax.set_xlabel("Goles")
    ax.set_ylabel("Probabilidad")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.bar(range(max_goles+1), p_vis)
    ax2.set_title("Distribución goles — Visitante")
    ax2.set_xlabel("Goles")
    ax2.set_ylabel("Probabilidad")
    st.pyplot(fig2)

    # Totals plot
    fig3, ax3 = plt.subplots()
    x = list(totals.keys())
    y = [totals[k] for k in x]
    ax3.bar(x, y)
    ax3.set_title("Probabilidad por total de goles")
    ax3.set_xlabel("Goles totales")
    ax3.set_ylabel("Probabilidad")
    st.pyplot(fig3)

    # Download CSV
    @st.cache_data
    def matrix_to_csv(df):
        return df.to_csv()

    csv = matrix_to_csv(df_matrix)
    st.download_button("Descargar matriz CSV", csv, file_name="matriz_marcadores.csv", mime="text/csv")

    st.markdown("**Notas:** Este es un modelo simplificado. Para mayor precisión puedes añadir factores: ventaja local, lesiones, xG, forma reciente, clima y ajustar λ con un factor de calibración.")