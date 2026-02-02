import tkinter

# Solo necesita importarse una vez al inicio de la aplicación
def apply_patch():
    # Parche (monkey patch) para el módulo interno _tkinter
    original_init = tkinter.Tk.__init__
    
    def patched_init(self, *args, **kwargs):
        # Llamar al init original
        original_init(self, *args, **kwargs)
        
        # Definir el procedimiento ::tk::ScreenChanged si no existe
        self.tk.eval("""
        if {[info commands ::tk::ScreenChanged] == ""} {
            proc ::tk::ScreenChanged {args} {
                # No hace nada
                return
            }
        }
        """)
    
    # Aplicar el parche
    tkinter.Tk.__init__ = patched_init

# Aplicar el parche automáticamente cuando se importe este módulo
apply_patch() 