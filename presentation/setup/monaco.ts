import { defineMonacoSetup } from '@slidev/types'
import { loadPyodide } from 'pyodide'

let pyodide: any = null
let initPromise: Promise<any> | null = null

export default defineMonacoSetup(() => {
  return {
    editorOptions: {
      theme: 'vitesse-dark',
    },
    customSetup(monaco) {
      return {
        runners: [
          {
            id: 'python',
            label: 'Python',
            run: async (code: string) => {
              try {
                if (!initPromise) {
                  initPromise = (async () => {
                    const py = await loadPyodide({
                      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
                    })
                    
                    // Provide visual feedback for long loading time
                    console.log("[Pyodide] Downloading Numpy, Pandas & micropip... this may take 10-20 seconds on first run.")
                    
                    // Load micropip to install our local wheel
                    await py.loadPackage(['micropip'])
                    const micropip = py.pyimport('micropip')
                    
                    // Install the parsfet wheel from the Slidev public directory
                    const wheelUrl = window.location.origin + '/parsfet-0.3.0a1-py3-none-any.whl'
                    await micropip.install(wheelUrl)

                    // Fetch the sample .lib file and write it to pyodide FS
                    console.log("[Pyodide] Injecting sample.lib to FileSystem")
                    const response = await fetch('/sample.lib')
                    const libContent = await response.text()
                    py.FS.writeFile('/sample.lib', libContent)
                    
                    console.log("[Pyodide] Environment ready!")
                    return py
                  })()
                }
                
                if (!pyodide) {
                  pyodide = await initPromise
                }
                
                // Redirect stdout to capture print statements
                let output = ''
                pyodide.setStdout({ batched: (str: string) => { output += str + '\n' } })
                
                // Run the code
                await pyodide.runPythonAsync(code)
                
                return {
                  text: output,
                  error: false,
                }
              } catch (e: any) {
                return {
                  text: e.message || String(e),
                  error: true,
                }
              }
            },
            sanitize(code: string) {
              return code
            },
          },
        ],
      }
    },
  }
})
