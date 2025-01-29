<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU IA avec Dataset</title>
    <script defer src="app.js"></script>
</head>
<body>
    <h1>WebGPU IA avec Dataset</h1>
    <input type="file" id="fileInput" />
    <canvas id="canvas" style="width: 500px; height: 500px;"></canvas>
</body>

<script>
    // Fonction pour lire le fichier texte et le transformer en dataset
    function readFile(file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const data = event.target.result;
            // Tokeniser les données
            const tokenizer = new Tokenizer();
            tokenizer.train(data);
            console.log("Vocabulaire créé:", tokenizer.vocab);

            // Ensuite, initialiser et entraîner le modèle
            initModelAndTrain(tokenizer);
        };
        reader.readAsText(file);
    }

    // Gérer l'upload du fichier
    document.getElementById('fileInput').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            readFile(file);
        }
    });

    // Classe Tokenizer
    class Tokenizer {
        constructor() {
            this.vocab = { "<PAD>": 0, "<UNK>": 1 };
            this.inverseVocab = ["<PAD>", "<UNK>"];
        }

        // Entraînement du tokenizer avec le corpus
        train(corpus) {
            const words = corpus.toLowerCase().split(/\s+/);
            words.forEach(word => {
                if (!this.vocab[word]) {
                    this.vocab[word] = this.inverseVocab.length;
                    this.inverseVocab.push(word);
                }
            });
        }

        // Encode le texte en liste d'IDs
        encode(text) {
            return text.toLowerCase().split(/\s+/).map(word => this.vocab[word] || this.vocab["<UNK>"]);
        }
    }

    // Initialisation du modèle WebGPU et démarrage de l'entraînement
    async function initModelAndTrain(tokenizer) {
        if (!navigator.gpu) {
            console.error("WebGPU n'est pas supporté !");
            return;
        }

        // Configurer WebGPU
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter.requestDevice();
        const context = document.getElementById("canvas").getContext("webgpu");
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format });

        // Créer un pipeline de rendu de base (placeholder pour un modèle IA)
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: device.createShaderModule({
                    code: ` @vertex fn main() -> @builtin(position) vec4<f32> { return vec4(0.0, 0.0, 0.0, 1.0); } `
                }),
                entryPoint: "main"
            },
            fragment: {
                module: device.createShaderModule({
                    code: ` @fragment fn main() -> @location(0) vec4<f32> { return vec4(1.0, 0.0, 0.0, 1.0); } `
                }),
                entryPoint: "main",
                targets: [{ format }]
            },
            primitive: { topology: "triangle-list" }
        });

        // Logique d'entraînement (simplifiée)
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: "clear",
                storeOp: "store",
                clearValue: [0.1, 0.1, 0.1, 1.0]
            }]
        });
        renderPass.setPipeline(pipeline);
        renderPass.draw(3);
        renderPass.end();

        // Soumettre les commandes
        device.queue.submit([commandEncoder.finish()]);

        // Afficher l'ID de tokenisation
        console.log("Texte tokenisé:", tokenizer.encode("exemple de texte"));
    }
</script>
</html>
