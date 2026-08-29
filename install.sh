#!/usr/bin/env bash
#
# Instalador de GIAR (asistente IA por consola).
#
# Uso:
#   curl -fsSL https://raw.githubusercontent.com/GIAR-UTN/GIAR-cli/main/install.sh | bash
#   o directamente:  bash install.sh
#
# Qué hace:
#   1. Comprueba Python >= 3.10.
#   2. Crea un entorno virtual aislado en $GIAR_HOME/venv (por defecto ~/.giar/venv).
#   3. Instala el paquete `giar-cli` desde PyPI dentro de ese entorno.
#   4. Crea el enlace `~/.local/bin/giar` apuntando al binario del entorno.
#   5. Añade ~/.local/bin al PATH (si hace falta) en tu shell de arranque.

set -euo pipefail

# ---- Colores ---------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

log()  { printf '%s\n' "${GREEN}==>${RESET} $*"; }
warn() { printf '%s\n' "${YELLOW}==>${RESET} $*"; }
die()  { printf '%s\n' "${RED}==>${RESET} $*" >&2; exit 1; }

# ---- Configuración ---------------------------------------------------------
VERSION="0.1.0"
# Fuente de instalación. Por defecto instala desde el repo de GitHub.
# Cuando el paquete esté publicado en PyPI, se puede usar:
#   GIAR_SOURCE="giar-cli" bash install.sh
# o fijar una versión concreta:
#   GIAR_SOURCE="giar-cli==0.1.0" bash install.sh
GIAR_SOURCE="${GIAR_SOURCE:-git+https://github.com/GIAR-UTN/GIAR-cli.git}"
GIAR_HOME="${GIAR_HOME:-$HOME/.giar}"
VENV_DIR="$GIAR_HOME/venv"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
BIN_LINK="$BIN_DIR/giar"

# ---- 1. Comprobar Python ----------------------------------------------------
check_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        die "No se encontró Python. Instala Python 3.10+ y vuelve a intentarlo."
    fi

    local ver
    ver="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$ver" < "3.10" ]]; then
        die "GIAR requiere Python 3.10+ (encontrado: $ver). Actualiza Python e inténtalo de nuevo."
    fi
    log "Python $ver encontrado: $("$PYTHON" --version 2>&1 | cut -d' ' -f2)"
}

# ---- 2. Crear entorno virtual -----------------------------------------------
setup_venv() {
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log "Creando entorno virtual en $VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
    else
        log "Entorno virtual ya existe en $VENV_DIR (se actualizará)"
    fi
}

# ---- 3. Instalar el paquete --------------------------------------------------
install_package() {
    if [[ "$GIAR_SOURCE" == git+* ]]; then
        log "Instalando GIAR $VERSION desde $GIAR_SOURCE"
        "$VENV_DIR/bin/pip" install --quiet --upgrade "$GIAR_SOURCE"
    else
        log "Instalando $GIAR_SOURCE desde PyPI"
        "$VENV_DIR/bin/pip" install --quiet --upgrade "$GIAR_SOURCE"
    fi
}

# ---- 4. Crear enlace en ~/.local/bin ------------------------------------------
link_binary() {
    mkdir -p "$BIN_DIR"
    if [[ -L "$BIN_LINK" ]] || [[ -f "$BIN_LINK" ]]; then
        rm -f "$BIN_LINK"
    fi
    ln -s "$VENV_DIR/bin/giar" "$BIN_LINK"
    chmod +x "$BIN_LINK"
    log "Enlace creado: $BIN_LINK"
}

# ---- 5. Añadir ~/.local/bin al PATH -------------------------------------------
ensure_path() {
    case ":$PATH:" in
        *":$BIN_DIR:"*) return ;;
    esac

    local rc=""
    for candidate in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [[ -f "$candidate" ]]; then rc="$candidate"; break; fi
    done
    [[ -z "$rc" ]] && rc="$HOME/.bashrc"

    # Evitar añadirlo dos veces
    if ! grep -qF "export PATH=\"$BIN_DIR:\$PATH\"" "$rc" 2>/dev/null; then
        printf '\n# Añadir %s al PATH (añadido por el instalador de GIAR)\nexport PATH="%s:$PATH"\n' "$BIN_DIR" "$BIN_DIR" >> "$rc"
        warn "$BIN_DIR no estaba en el PATH. Añadido a $rc"
    fi
    warn "Abre una nueva terminal (o ejecuta: source $rc) para usar 'giar'."
}

# ---- Resumen ------------------------------------------------------------------
finish() {
    log "Instalación completada."
    printf '\n%s%s  GIAR %s  %s\n' "$BOLD" "$GREEN" "$VERSION" "$RESET"
    printf '%s\n' "-----------------------------------------"
    printf '  Ejecuta:  %sgiar%s           -> chat interactivo\n' "$BOLD" "$RESET"
    printf '            %sgiar doctor%s    -> diagnóstico\n' "$BOLD" "$RESET"
    printf '            %sgiar config llm%s -> configurar el modelo\n' "$BOLD" "$RESET"
    printf '\n%sTodo se guarda en %s%s\n' "$DIM" "$GIAR_HOME" "$RESET"
}

# ---- Main -----------------------------------------------------------------------
main() {
    check_python
    setup_venv
    install_package
    link_binary
    ensure_path
    finish
}

main "$@"
