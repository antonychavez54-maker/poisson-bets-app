# Poisson Bets — Mini App

Pequeña app en Streamlit que implementa un modelo Poisson simplificado para estimar probabilidades de goles,
BTTS, Over/Under y marcador más probable en partidos de fútbol.

## Archivos incluidos
- `app.py` : Código principal de la app Streamlit.
- `requirements.txt` : Dependencias Python.
- `Procfile` : Instrucción para plataformas tipo Heroku/Railway/Render.
- `runtime.txt` : Versión de Python (opcional para Heroku).
- `README.md` : Este archivo.

## Ejecutar localmente
1. Crear y activar un entorno virtual:
   - Windows (PowerShell):
     ```
     python -m venv venv
     .\venv\Scripts\Activate
     ```
   - macOS / Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Ejecutar:
   ```
   streamlit run app.py
   ```

## Despliegue rápido
- **Streamlit Cloud**: subir repo a GitHub y conectar en https://share.streamlit.io
- **Railway / Render / Heroku**: subir repo a GitHub y usar `Procfile`. Comando de inicio incluido en `Procfile`.

## Mejoras sugeridas
- Conectar con una API deportiva para obtener automáticamente promedios/xG.
- Añadir factor de ventaja local y calibración.
- Guardar historial y comparar predicción vs resultado real.