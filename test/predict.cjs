var { FastText, init } = require("../dist/xtranslator.es.js");
const fs = require("node:fs");

const printVector = function (predictions, limit) {
	limit = limit || Infinity;

	for (let i = 0; i < predictions.size() && i < limit; i++) {
		let prediction = predictions.get(i);
		console.log(predictions.get(i));
	}
};

init({ wasm: fs.readFileSync("../assets/fasttext_wasm.wasm") }).then(() => {
	let ft = new FastText();

	const url = "lid.176.ftz";
	ft.loadModel(fs.readFileSync("../assets/lid.176.ftz")).then((model) => {
		let text = "Bonjour à tous. Ceci est du français";
		console.log(text);
		printVector(model.predict(text, 5, 0.0));

		text = "Hello, world. This is english";
		console.log(text);
		printVector(model.predict(text, 5, 0.0));

		text = "Merhaba dünya. Bu da türkçe";
		console.log(text);
		printVector(model.predict(text, 5, 0.0));
	});
});
