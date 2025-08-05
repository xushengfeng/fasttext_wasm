const { initFastText } = require("../dist/fasttext.es.js");

const printVector = (predictions) => {
    const limit = Number.POSITIVE_INFINITY;

    for (let i = 0; i < predictions.size() && i < limit; i++) {
        console.log(predictions.get(i));
    }
};

initFastText().then((x) => {
    const ft = x.FastText();
    ft.loadModel().then((model) => {
        let text = "Bonjour à tous. Ceci est du français";
        console.log(text);
        printVector(model.predict(text, 5, 0.0));
        console.log(model.identify(text));

        text = "Hello, world. This is english";
        console.log(text);
        printVector(model.predict(text, 5, 0.0));

        text = "Merhaba dünya. Bu da türkçe";
        console.log(text);
        printVector(model.predict(text, 5, 0.0));
    });
});
