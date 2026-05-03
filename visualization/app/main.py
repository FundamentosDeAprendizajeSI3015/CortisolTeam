"""
Script Principal — Genera todas las visualizaciones y dashboard
Crypto ML Project — Visualizacion Interactiva

Uso:
    python visualization/app/main.py

Ejecucion:
    1. Lee datos de EDA y clustering
    2. Genera todas las graficas PNG
    3. Crea dashboard HTML interactivo
    4. Abre el dashboard en el navegador
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import time


def resolver_python_ejecutable() -> Path:
    """
    Busca el Python del entorno virtual del workspace (ml_env) y, si no existe,
    usa el interprete actual.

    Returns:
        Ruta al ejecutable de Python a utilizar.
    """
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "ml_env" / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def ejecutar_script(script_path: Path, nombre: str, python_exec: Path) -> bool:
    """
    Ejecuta un script Python y captura su salida.
    
    Args:
        script_path: Ruta del script a ejecutar
        nombre: Nombre descriptivo del script
        python_exec: Ejecutable de Python a usar
        
    Returns:
        True si la ejecucion fue exitosa, False en caso contrario
    """
    if not script_path.exists():
        print(f"[ERROR] No se encontro: {script_path}")
        return False
    
    print(f"\n[EJECUCION] Iniciando: {nombre}...\n")
    
    try:
        resultado = subprocess.run([str(python_exec), str(script_path)],
                                 capture_output=False, text=True)
        
        if resultado.returncode != 0:
            print(f"[ERROR] {nombre} fallo con codigo: {resultado.returncode}")
            return False
        
        return True
    
    except Exception as e:
        print(f"[ERROR] No se pudo ejecutar {nombre}: {e}")
        return False


def main():
    """
    Funcion principal que orquesta todo el flujo.
    """
    print("\n" + "="*70)
    print(" PIPELINE COMPLETO DE VISUALIZACIONES")
    print(" Crypto ML Project")
    print("="*70)
    
    # Rutas
    app_dir = Path(__file__).resolve().parent
    visualization_dir = app_dir.parent
    output_dir = visualization_dir / "outputs"
    visualize_script = app_dir / "visualize.py"
    dashboard_script = app_dir / "generate_dashboard.py"
    dashboard_html = app_dir / "dashboard.html"
    python_exec = resolver_python_ejecutable()

    print(f"\n[PYTHON] Ejecutable usado: {python_exec}")
    
    # Paso 1: Generar visualizaciones
    print("\n[PASO 1/3] Generando visualizaciones PNG...")
    if not ejecutar_script(visualize_script, "Visualizaciones", python_exec):
        print("\n[ERROR] Fallo en la generacion de visualizaciones.")
        sys.exit(1)
    
    # Paso 2: Generar dashboard HTML
    print("\n[PASO 2/3] Generando dashboard HTML...")
    if not ejecutar_script(dashboard_script, "Dashboard HTML", python_exec):
        print("\n[ERROR] Fallo en la generacion del dashboard.")
        sys.exit(1)
    
    # Paso 3: Abrir dashboard en navegador
    print("\n[PASO 3/3] Abriendo dashboard en navegador...")
    time.sleep(1)  # Esperar un poco para asegurar que el archivo este listo
    
    if dashboard_html.exists():
        try:
            dashboard_url = dashboard_html.as_uri()
            webbrowser.open(dashboard_url)
            print(f"[OK] Dashboard abierto en navegador")
        except Exception as e:
            print(f"[AVISO] No se pudo abrir automaticamente: {e}")
            print(f"[INFO] Abre manualmente: {dashboard_html}")
    else:
        print(f"[ERROR] No se encontro el dashboard: {dashboard_html}")
        sys.exit(1)
    
    # Resumen final
    print("\n" + "="*70)
    print(" COMPLETADO: Pipeline de visualizaciones terminado.")
    print("="*70)
    print(f"\n Visualizaciones PNG en: {output_dir}")
    print(f" Dashboard interactivo en: {dashboard_html}")
    print("\n Puedes volver a abrir el dashboard en cualquier momento con:")
    print(f" {dashboard_url}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
