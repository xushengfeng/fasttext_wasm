/// <reference types="emscripten" />

interface Vector<T> {
    clone: () => Vector<T>;
    delete: () => void;
    get: (index: number) => T;
    push_back: (value: T) => void;
    resize: (size: number, value?: T) => void;
    set: (index: number, value: T) => void;
    size: () => number;
}

type Pair<T1, T2> = [T1, T2];

type int = number;

type DenseMatrix = {
    rows: number;
    cols: number;
    at: (i: number, j: number) => number;
};

export interface FastTextCoreConstructor {
    new (): FastTextCore;
    loadModel: (path: string) => void;
    getNN: (word: string, k: number) => Vector<Pair<number, string>>;
    getAnalogies: (
        k: number,
        wordA: string,
        wordB: string,
        wordC: string,
    ) => Vector<Pair<number, string>>;
    getWordId: (word: string) => number;
    getSubwordId: (subword: string) => number;
    getInputMatrix: () => DenseMatrix;
    getOutputMatrix: () => DenseMatrix;
    getWords: () => Pair<Vector<string>, Vector<number>>;
    getLabels: () => Pair<Vector<string>, Vector<number>>;
    getLine: (text: string) => Pair<Vector<string>, Vector<string>>;
    test: (filename: string, k: number, thresold: number) => void;
    predict: (
        text: string,
        k: number,
        thresold: number,
    ) => Vector<Pair<number, string>>;
    getWordVector: (vecFloat: Float32ArrayBridge, word: string) => void;
    getSentenceVector: (vecFloat: Float32ArrayBridge, text: string) => void;
    getInputVector: (vecFloat: Float32ArrayBridge, ind: number) => void;
    train: (
        args: unknown,
        jsCallback:
            | ((
                  progress: number,
                  loss: number,
                  wst: number,
                  lr: number,
                  eta: number,
              ) => void)
            | null,
    ) => void;
    saveModel: (filename: string) => void;
    isQuant: boolean;

    args: any;
    getSubwords: (word: string) => Pair<string[], number[]>;
}

export type FastTextCore = FastTextCoreConstructor;

export interface Float32ArrayBridge {
    ptr: number;
    size: number;
}

const fs = FS;
type TFS = typeof fs;

export interface FastTextModule extends EmscriptenModule {
    FS: {
        writeFile: TFS["writeFile"];
        readFile: TFS["readFile"];
        unlink: TFS["unlink"];
    };
    FastText: FastTextCoreConstructor;
    Args: any;
    ModelName: any;
    LossName: any;
}
