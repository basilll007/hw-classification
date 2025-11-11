async function sendPrediction() {
  const payload = {
    Gene_E_Housekeeping: parseFloat(document.getElementById("g1").value),
    Gene_A_Oncogene: parseFloat(document.getElementById("g2").value),
    Gene_B_Immune: parseFloat(document.getElementById("g3").value),
    Gene_C_Stromal: parseFloat(document.getElementById("g4").value),
    Gene_D_Therapy: parseFloat(document.getElementById("g5").value),
    Pathway_Score_Inflam: parseFloat(document.getElementById("g6").value),
    UMAP_1: parseFloat(document.getElementById("g7").value),
    Disease_Status: document.getElementById("status").value // accepts 'Tumor' or 'Normal'
  };

  try {
    const r = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Request failed");
    const probs = j.probabilities.map((p) => p.toFixed(3));
    document.getElementById("result").innerText =
      `Prediction: ${j.prediction}\nClasses: ${j.classes.join(", ")}\nProbabilities: ${probs.join(", ")}`;
  } catch (e) {
    document.getElementById("result").innerHTML = `<span class="bad">Error: ${e.message}</span>`;
  }
}
