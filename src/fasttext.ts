/// <reference types="vite/client" />
import fastTextModularized from "./fasttext_wasm.js";
import type {
    FastTextCore,
    FastTextModule,
    int,
    Pair,
    Vector,
} from "./fastText.d.ts";
import lanJson from "./languages.json?raw";
const lanJ = JSON.parse(lanJson) as Record<
    string,
    { alpha3: string; alpha2: string | null; refName: string }
>;

const _initFastTextModule: (options?: {
    wasm?: ArrayBuffer;
}) => Promise<{ FastText: () => FastText }> = async (x) => {
    const fastTextModule = await fastTextModularized({
        wasmBinary:
            x?.wasm ??
            (await import("../assets/fasttext_wasm.wasm?binary")).default,
    });
    return { FastText: () => new FastText(fastTextModule) };
};

const modelFileInWasmFs = "model.bin";

const getFloat32ArrayFromHeap = (
    len: number,
    fastTextModule: FastTextModule,
) => {
    const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
    const dataPtr = fastTextModule._malloc(dataBytes);
    const dataHeap = new Uint8Array(
        fastTextModule.HEAPU8.buffer,
        dataPtr,
        dataBytes,
    );
    return {
        ptr: dataHeap.byteOffset,
        size: len,
        buffer: dataHeap.buffer,
    };
};

const heapToFloat32 = (r) => new Float32Array(r.buffer, r.ptr, r.size);

class FastText {
    fastTextModule: FastTextModule;
    f: FastTextCore;
    constructor(fastTextModule: FastTextModule) {
        this.fastTextModule = fastTextModule;
        this.f = new this.fastTextModule.FastText();
    }
    async loadModel(bytes?: ArrayBuffer) {
        const fastTextNative = this.f;
        const byteArray = new Uint8Array(
            bytes ?? (await import("../assets/lid.176.ftz?binary")).default,
        );
        const FS = this.fastTextModule.FS;
        FS.writeFile(modelFileInWasmFs, byteArray);
        fastTextNative.loadModel(modelFileInWasmFs);
        return new FastTextModel(fastTextNative, this.fastTextModule);
    }
}

class FastTextModel {
    fastTextModule: FastTextModule;
    f: FastTextCore;
    constructor(fastTextNative: FastTextCore, fastTextModule: FastTextModule) {
        this.f = fastTextNative;
        this.fastTextModule = fastTextModule;
    }

    /**
     * isQuant
     *
     * @return true if the model is quantized
     *
     */
    isQuant() {
        return this.f.isQuant;
    }

    /**
     * getDimension
     *
     * @return   the dimension (size) of a lookup vector (hidden layer)
     *
     */
    getDimension(): int {
        return this.f.args.dim;
    }

    /**
     * getWordVector
     *
     * @param     word
     *
     * @return   the vector representation of `word`.
     *
     */
    getWordVector(word: string): Float32Array {
        const b = getFloat32ArrayFromHeap(
            this.getDimension(),
            this.fastTextModule,
        );
        this.f.getWordVector(b, word);

        return heapToFloat32(b);
    }

    /**
     * getSentenceVector
     *
     * @param          text
     *
     * @return   the vector representation of `text`.
     *
     */
    getSentenceVector(_text: string): Float32Array {
        let text = _text;
        if (text.indexOf("\n") !== -1) {
            ("sentence vector processes one line at a time (remove '\\n')");
        }
        text += "\n";
        const b = getFloat32ArrayFromHeap(
            this.getDimension(),
            this.fastTextModule,
        );
        this.f.getSentenceVector(b, text);

        return heapToFloat32(b);
    }

    /**
     * getNearestNeighbors
     *
     * returns the nearest `k` neighbors of `word`.
     *
     * @param {string}          word
     * @param {int}             k
     *
     * @return
     *     words and their corresponding cosine similarities.
     *
     */
    getNearestNeighbors(
        word: string,
        k: int = 10,
    ): Vector<Pair<number, string>> {
        return this.f.getNN(word, k);
    }

    /**
     * getAnalogies
     *
     * returns the nearest `k` neighbors of the operation
     * `wordA - wordB + wordC`.
     *
     * @param {string}          wordA
     * @param {string}          wordB
     * @param {string}          wordC
     * @param {int}             k
     *
     * @return
     *     words and their corresponding cosine similarities
     *
     */
    getAnalogies(
        wordA: string,
        wordB: string,
        wordC: string,
        k: int,
    ): Vector<Pair<number, string>> {
        return this.f.getAnalogies(k, wordA, wordB, wordC);
    }

    /**
     * getWordId
     *
     * Given a word, get the word id within the dictionary.
     * Returns -1 if word is not in the dictionary.
     *
     * @return {int}    word id
     *
     */
    getWordId(word: string): int {
        return this.f.getWordId(word);
    }

    /**
     * getSubwordId
     *
     * Given a subword, return the index (within input matrix) it hashes to.
     *
     * @return {int}    subword id
     *
     */
    getSubwordId(subword: string): int {
        return this.f.getSubwordId(subword);
    }

    /**
     * getSubwords
     *
     * returns the subwords and their indicies.
     *
     * @param {string}          word
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     words and their corresponding indicies
     *
     */
    getSubwords(word: string): Pair<Array<string>, Array<int>> {
        return this.f.getSubwords(word);
    }

    /**
     * getInputVector
     *
     * Given an index, get the corresponding vector of the Input Matrix.
     *
     * @param {int}             ind
     *
     * @return {Float32Array}   the vector of the `ind`'th index
     *
     */
    getInputVector(ind: int) {
        const b = getFloat32ArrayFromHeap(
            this.getDimension(),
            this.fastTextModule,
        );
        this.f.getInputVector(b, ind);

        return heapToFloat32(b);
    }

    /**
     * predict
     *
     * Given a string, get a list of labels and a list of corresponding
     * probabilities. k controls the number of returned labels.
     *
     * @param {string}          text
     * @param {int}             k, the number of predictions to be returned
     * @param {number}          probability threshold
     *
     * @return
     *     labels and their probabilities
     *
     */
    predict(text: string, k: int = 1, threshold = 0.0) {
        return this.f.predict(text, k, threshold);
    }

    /**
     * getInputMatrix
     *
     * Get a reference to the full input matrix of a Model. This only
     * works if the model is not quantized.
     *
     * @return
     *     densematrix with functions: `rows`, `cols`, `at(i,j)`
     *
     * example:
     *     let inputMatrix = model.getInputMatrix();
     *     let value = inputMatrix.at(1, 2);
     */
    getInputMatrix() {
        if (this.isQuant()) {
            throw new Error("Can't get quantized Matrix");
        }
        return this.f.getInputMatrix();
    }

    /**
     * getOutputMatrix
     *
     * Get a reference to the full input matrix of a Model. This only
     * works if the model is not quantized.
     *
     * @return
     *     densematrix with functions: `rows`, `cols`, `at(i,j)`
     *
     * example:
     *     let outputMatrix = model.getOutputMatrix();
     *     let value = outputMatrix.at(1, 2);
     */
    getOutputMatrix() {
        if (this.isQuant()) {
            throw new Error("Can't get quantized Matrix");
        }
        return this.f.getOutputMatrix();
    }

    /**
     * getWords
     *
     * Get the entire list of words of the dictionary including the frequency
     * of the individual words. This does not include any subwords. For that
     * please consult the function get_subwords.
     *
     * @return
     *     words and their corresponding frequencies
     *
     */
    getWords() {
        return this.f.getWords();
    }

    /**
     * getLabels
     *
     * Get the entire list of labels of the dictionary including the frequency
     * of the individual labels.
     *
     * @return
     *     labels and their corresponding frequencies
     *
     */
    getLabels() {
        return this.f.getLabels();
    }

    /**
     * getLine
     *
     * Split a line of text into words and labels. Labels must start with
     * the prefix used to create the model (__label__ by default).
     *
     * @param {string}          text
     *
     * @return
     *     words and labels
     *
     */
    getLine(text: string) {
        return this.f.getLine(text);
    }
    identify(text: string) {
        const predictions = this.predict(text);
        const l = predictions.get(0)[1].slice("__label__".length);
        if (l) {
            const lj = lanJ[l];
            return lj;
        }
        return undefined;
    }
}

export { _initFastTextModule as initFastText };
