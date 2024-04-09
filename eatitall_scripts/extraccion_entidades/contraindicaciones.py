import sys
sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import json
import re

def clasificar_elementos(texto, frases_indicaciones=[], frases_contraindicaciones=[]):
    if frases_indicaciones==[]:
        frases_indicaciones=["aumentar","consumir"]
    if frases_contraindicaciones==[]:
        frases_contraindicaciones=["eliminar", "sustituir", "reducir"]

    # Dividir el texto en frases para procesar cada recomendación por separado
    frases = re.split(r'\.\s*|\n', texto)
    
    # Listas para almacenar las oraciones de indicaciones y contraindicaciones
    elementos_indicaciones = []
    elementos_contraindicaciones = []
    
    # Procesar cada frase para clasificar
    for frase in frases:
        # Limpiar la frase antes de procesarla
        frase_limpia = re.sub(r'^[^A-Za-z]+', '', frase)
        
        # Determinar si la frase es de indicación o contraindicación y limpiarla
        for opcion in frases_indicaciones:
            if frase_limpia.startswith(opcion):
                frase_limpia = frase_limpia.replace(opcion, '').strip()
                elementos_indicaciones.append(frase_limpia)
                break

        for opcion in frases_contraindicaciones:
            if frase_limpia.startswith(opcion):
                frase_limpia = frase_limpia.replace(opcion, '').strip()
                elementos_contraindicaciones.append(frase_limpia)
                break
    
    # Devolver los resultados
    return elementos_indicaciones, elementos_contraindicaciones