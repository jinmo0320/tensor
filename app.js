const table = document.getElementById("resultTable");
const epochInput = document.getElementById("epochInput");
const fitBtn = document.getElementById("fitBtn");
const fitLog = document.getElementById("fitLog");
const modelDiv = document.getElementById("modelDiv");
const preInput = document.getElementById("preInput");
const preBtn = document.getElementById("preBtn");
const preDiv = document.getElementById("preDiv");

const 온도 = [20, 21, 22, 23, 24];
const 판매량 = [40, 42, 45, 47, 49];

const 원인 = tf.tensor(온도);
const 결과 = tf.tensor(판매량);

const x = tf.input({ shape: [1] });
const y = tf.layers.dense({ units: 1 }).apply(x);
const model = tf.model({ inputs: x, outputs: y });
const compileParam = {
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError,
};
model.compile(compileParam);

fitBtn.onclick = () => {
  const fitParam = {
    epochs: epochInput.value,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        fitLog.innerHTML = `학습 횟수: ${
          epoch + 1
        } <br> 학습률(오차): ${Math.sqrt(logs.loss).toFixed(2)}`;
      },
    },
  };

  model.fit(원인, 결과, fitParam).then((result) => {
    const weights = model.getWeights();
    const weight = weights[0].arraySync()[0][0];
    const bias = weights[1].arraySync()[0];

    table.innerHTML = `
    <th>온도</th>
    <th>판매량</th>
    <tr>
      <td>20</td>
      <td>${model
        .predict(tf.tensor([20]))
        .arraySync()[0][0]
        .toFixed(2)}</td>
    </tr>
    <tr>
      <td>21</td>
      <td>${model
        .predict(tf.tensor([21]))
        .arraySync()[0][0]
        .toFixed(2)}</td>
    </tr>
    <tr>
      <td>22</td>
      <td>${model
        .predict(tf.tensor([22]))
        .arraySync()[0][0]
        .toFixed(2)}</td>
    </tr>
    <tr>
      <td>23</td>
      <td>${model
        .predict(tf.tensor([23]))
        .arraySync()[0][0]
        .toFixed(2)}</td>
    </tr>
    <tr>
      <td>24</td>
      <td>${model
        .predict(tf.tensor([24]))
        .arraySync()[0][0]
        .toFixed(2)}</td>
    </tr>
    `;
    modelDiv.innerHTML = `y = ${weight.toFixed(2)} x + ${bias.toFixed(2)}`;
  });
};

preBtn.onclick = () => {
  const preResult = model
    .predict(tf.tensor([parseInt(preInput.value)]))
    .arraySync()[0][0];
  preDiv.innerHTML = `예측값: ${preResult.toFixed(2)}`;
};
