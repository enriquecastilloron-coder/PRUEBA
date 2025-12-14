# 🚀 GUÍA DE DESPLIEGUE STREAMLIT - PASO A PASO

## ✅ LO QUE TIENES

Has recibido 5 archivos esenciales para tu aplicación Streamlit:

1. **app.py** - Aplicación principal
2. **requirements.txt** - Dependencias de Python
3. **README.md** - Documentación completa
4. **.gitignore** - Archivos a ignorar en Git
5. **holmen_example.csv** - Datos de ejemplo
6. **.streamlit/config.toml** - Configuración de tema

---

## 🎯 OPCIÓN 1: DESPLIEGUE EN STREAMLIT CLOUD (RECOMENDADO)

### **GRATIS y PERMANENTE** ⭐

### Paso 1: Crear repositorio en GitHub

1. Ve a https://github.com/EnriqueCastilloRon
2. Click en **"New repository"** (botón verde)
3. Nombre: `fatigue-analysis-app`
4. Descripción: `Fatigue Analysis Web Application with PyMC and Streamlit`
5. ✅ Marca "Public"
6. ✅ Marca "Add README file"
7. Click **"Create repository"**

### Paso 2: Subir archivos

1. En tu nuevo repositorio, click **"Add file"** → **"Upload files"**
2. Arrastra TODOS los archivos que descargaste:
   - app.py
   - requirements.txt
   - README.md
   - .gitignore
   - holmen_example.csv
   - .streamlit/config.toml (crea carpeta .streamlit y sube config.toml dentro)
3. Escribe mensaje: `Initial commit - Fatigue analysis app`
4. Click **"Commit changes"**

### Paso 3: Desplegar en Streamlit Cloud

1. Ve a **https://share.streamlit.io**
2. Click **"Sign in with GitHub"**
3. Autoriza Streamlit
4. Click **"New app"**
5. Selecciona:
   - Repository: `EnriqueCastilloRon/fatigue-analysis-app`
   - Branch: `main`
   - Main file path: `app.py`
6. Click **"Deploy!"**

### Paso 4: ¡Listo!

Tu aplicación estará en:
```
https://enriquecastilloron-fatigue-analysis-app.streamlit.app
```

⏱️ El despliegue toma 2-5 minutos la primera vez.

---

## 🎯 OPCIÓN 2: EJECUTAR LOCALMENTE (PARA PROBAR)

### Paso 1: Instalar Python

Si no tienes Python instalado:
- Windows: https://www.python.org/downloads/
- Mac: `brew install python3`
- Linux: `sudo apt install python3`

### Paso 2: Instalar dependencias

```bash
# Navega a la carpeta con los archivos
cd /ruta/a/tus/archivos

# Instala las dependencias
pip install -r requirements.txt
```

### Paso 3: Ejecutar

```bash
streamlit run app.py
```

Tu navegador abrirá automáticamente en `http://localhost:8501`

---

## 🎯 OPCIÓN 3: HUGGINGFACE SPACES (ALTERNATIVA)

1. Crea cuenta en https://huggingface.co
2. Click **"New Space"**
3. Nombre: `fatigue-analysis`
4. SDK: Selecciona **Streamlit**
5. Sube `app.py` y `requirements.txt`
6. ¡Listo! Tu app estará en:
   ```
   https://huggingface.co/spaces/TU_USERNAME/fatigue-analysis
   ```

---

## 📊 CÓMO USAR LA APLICACIÓN

### 1. Pestaña "Data" (Datos)
- Elige "Holmen Example" o sube tu propio archivo
- Formatos soportados: CSV, Excel, TXT (formato R)
- Click "Load" / "Process"

### 2. Pestaña "Analysis" (Análisis)
- Click **"Run MLE"** (estimación inicial)
- Espera resultados
- Click **"Run MCMC"** (inferencia Bayesiana)
- Espera ~2-5 minutos

### 3. Pestaña "Results" (Resultados)
- Ve resumen posterior
- Gráficos de diagnóstico
- Click **"Compute Percentile Curves"** para curvas S-N

### 4. Pestaña "About"
- Documentación completa
- Referencias

---

## ⚙️ CONFIGURACIÓN (Sidebar)

Ajusta parámetros antes de ejecutar:

- **MCMC Settings**:
  - Warmup: 1000 (recomendado)
  - Draws: 2000 (recomendado)
  - Chains: 2 (mínimo para convergencia)
  - Target accept: 0.95

- **Percentile Settings**:
  - Stress min/max: Rango de tensiones
  - Points: Resolución de curvas
  - Samples: Muestras para percentiles

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Error: "ModuleNotFoundError"
```bash
pip install --upgrade -r requirements.txt
```

### MCMC muy lento
- Reduce "Draws" a 1000
- Reduce "Chains" a 1 (solo para pruebas)

### Aplicación no carga
- Verifica que todos los archivos estén en el repositorio
- Revisa los logs en Streamlit Cloud

---

## 📞 SOPORTE

Si tienes problemas:
1. Revisa README.md
2. Consulta https://docs.streamlit.io
3. Abre issue en tu repositorio de GitHub

---

## 🎉 ¡FELICIDADES!

Ahora tienes una **aplicación web profesional** para análisis de fatiga:

✅ Interfaz interactiva moderna
✅ Análisis Bayesiano completo
✅ Gráficos profesionales
✅ Hosting gratuito permanente
✅ URL pública para compartir

**Comparte tu URL con colegas y publica tus resultados!**

---

Creado con ❤️ usando Streamlit, PyMC, y ArviZ
