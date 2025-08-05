# fastText wasm

fastText 是 Facebook 开发的文本语言识别工具。在用于翻译语言检测时非常有用。

通过 es6 import 语法引入 wasm 和模型，不需要 `fs` 或 `fetch`，在 Electron 环境中特别有用。

缺省模型：`lid.176.ftz`

```js
const { initFastText } = require("../dist/fasttext.es.js");

initFastText().then((x) => {
    // load wasm
    const ft = x.FastText();
    ft.loadModel().then((model) => {
        const text = "Hello, world. This is english";
        console.log(model.identify(text));
        // { alpha3: 'eng', alpha2: 'en', refName: 'English' }
    });
});
```

此库仅用于推理，不用于训练。
