import express from "express";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

app.use(express.json({ limit: "25mb" }));

// Helper to execute Python prediction script
function runPythonPrediction(specs: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = execFile(
      "python3",
      ["predict_cli.py", JSON.stringify(specs)],
      { cwd: __dirname },
      (error, stdout, stderr) => {
        if (error) {
          console.error("Prediction error:", stderr);
          return reject(new Error(stderr || error.message));
        }
        try {
          const res = JSON.parse(stdout.trim());
          resolve(res);
        } catch (e) {
          reject(new Error(`Failed to parse prediction output: ${stdout}`));
        }
      }
    );
  });
}

// Helper to execute Python training script
function runPythonTraining(): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile("python3", ["train_now.py"], { cwd: __dirname }, (error, stdout, stderr) => {
      if (error) {
        console.error("Training error:", stderr);
        return reject(new Error(stderr || error.message));
      }
      resolve(stdout);
    });
  });
}

// 1. Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// 2. Dataset retrieval
app.get("/api/dataset", (req, res) => {
  try {
    const laptopsPath = path.join(__dirname, "models", "cleaned_laptops.json");
    if (!fs.existsSync(laptopsPath)) {
      return res.status(404).json({ error: "Cleaned dataset not found. Please train models first." });
    }
    const laptops = JSON.parse(fs.readFileSync(laptopsPath, "utf-8"));
    
    // Compute quick stats
    const prices = laptops.map((l: any) => l.Price);
    const brands = Array.from(new Set(laptops.map((l: any) => l.Brand)));
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const avgPrice = Math.round(prices.reduce((a: number, b: number) => a + b, 0) / prices.length);

    res.json({
      laptops,
      stats: {
        total_count: laptops.length,
        brands_count: brands.length,
        brands,
        min_price: minPrice,
        max_price: maxPrice,
        avg_price: avgPrice,
      }
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 3. Model metrics & comparison
app.get("/api/models", (req, res) => {
  try {
    const metricsPath = path.join(__dirname, "models", "model_metrics.json");
    if (!fs.existsSync(metricsPath)) {
      return res.status(404).json({ error: "Model metrics not found." });
    }
    const metrics = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
    res.json(metrics);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 4. EDA Analytics
app.get("/api/eda", (req, res) => {
  try {
    const laptopsPath = path.join(__dirname, "models", "cleaned_laptops.json");
    if (!fs.existsSync(laptopsPath)) {
      return res.status(404).json({ error: "Dataset not found." });
    }
    const laptops: any[] = JSON.parse(fs.readFileSync(laptopsPath, "utf-8"));

    // Brand aggregations
    const brandMap: Record<string, { count: number; sum: number; min: number; max: number }> = {};
    for (const l of laptops) {
      if (!brandMap[l.Brand]) {
        brandMap[l.Brand] = { count: 0, sum: 0, min: Infinity, max: -Infinity };
      }
      brandMap[l.Brand].count += 1;
      brandMap[l.Brand].sum += l.Price;
      if (l.Price < brandMap[l.Brand].min) brandMap[l.Brand].min = l.Price;
      if (l.Price > brandMap[l.Brand].max) brandMap[l.Brand].max = l.Price;
    }
    const brandStats = Object.keys(brandMap).map(brand => ({
      brand,
      count: brandMap[brand].count,
      avg_price: Math.round(brandMap[brand].sum / brandMap[brand].count),
      min_price: brandMap[brand].min,
      max_price: brandMap[brand].max,
    })).sort((a, b) => b.avg_price - a.avg_price);

    // RAM aggregations
    const ramMap: Record<string, { count: number; sum: number }> = {};
    for (const l of laptops) {
      const k = String(l.RAM);
      if (!ramMap[k]) ramMap[k] = { count: 0, sum: 0 };
      ramMap[k].count += 1;
      ramMap[k].sum += l.Price;
    }
    const ramStats = Object.keys(ramMap)
      .map(ram => ({ ram: Number(ram), count: ramMap[ram].count, avg_price: Math.round(ramMap[ram].sum / ramMap[ram].count) }))
      .sort((a, b) => a.ram - b.ram);

    // Storage aggregations
    const storageMap: Record<string, { count: number; sum: number }> = {};
    for (const l of laptops) {
      const k = String(l.Storage);
      if (!storageMap[k]) storageMap[k] = { count: 0, sum: 0 };
      storageMap[k].count += 1;
      storageMap[k].sum += l.Price;
    }
    const storageStats = Object.keys(storageMap)
      .map(storage => ({ storage: Number(storage), count: storageMap[storage].count, avg_price: Math.round(storageMap[storage].sum / storageMap[storage].count) }))
      .sort((a, b) => a.storage - b.storage);

    // Processor tier aggregations
    const procMap: Record<string, { count: number; sum: number }> = {};
    for (const l of laptops) {
      const k = l.Processor_Tier || "Other";
      if (!procMap[k]) procMap[k] = { count: 0, sum: 0 };
      procMap[k].count += 1;
      procMap[k].sum += l.Price;
    }
    const procStats = Object.keys(procMap)
      .map(tier => ({ tier, count: procMap[tier].count, avg_price: Math.round(procMap[tier].sum / procMap[tier].count) }))
      .sort((a, b) => b.avg_price - a.avg_price);

    // GPU tier aggregations
    const gpuMap: Record<string, { count: number; sum: number }> = {};
    for (const l of laptops) {
      const k = l.GPU_Tier || "Integrated";
      if (!gpuMap[k]) gpuMap[k] = { count: 0, sum: 0 };
      gpuMap[k].count += 1;
      gpuMap[k].sum += l.Price;
    }
    const gpuStats = Object.keys(gpuMap)
      .map(tier => ({ tier, count: gpuMap[tier].count, avg_price: Math.round(gpuMap[tier].sum / gpuMap[tier].count) }))
      .sort((a, b) => b.avg_price - a.avg_price);

    // OS aggregations
    const osMap: Record<string, { count: number; sum: number }> = {};
    for (const l of laptops) {
      const k = l.Operating_System || "Other";
      if (!osMap[k]) osMap[k] = { count: 0, sum: 0 };
      osMap[k].count += 1;
      osMap[k].sum += l.Price;
    }
    const osStats = Object.keys(osMap)
      .map(os => ({ os, count: osMap[os].count, avg_price: Math.round(osMap[os].sum / osMap[os].count) }))
      .sort((a, b) => b.avg_price - a.avg_price);

    // Price brackets
    const brackets = [
      { label: "< ₹35k", min: 0, max: 35000, count: 0 },
      { label: "₹35k - ₹50k", min: 35000, max: 50000, count: 0 },
      { label: "₹50k - ₹75k", min: 50000, max: 75000, count: 0 },
      { label: "₹75k - ₹1L", min: 75000, max: 100000, count: 0 },
      { label: "₹1L - ₹1.5L", min: 100000, max: 150000, count: 0 },
      { label: "₹1.5L - ₹2L", min: 150000, max: 200000, count: 0 },
      { label: "> ₹2L", min: 200000, max: Infinity, count: 0 },
    ];
    for (const l of laptops) {
      for (const b of brackets) {
        if (l.Price >= b.min && l.Price < b.max) {
          b.count += 1;
          break;
        }
      }
    }

    res.json({
      brandStats,
      ramStats,
      storageStats,
      procStats,
      gpuStats,
      osStats,
      brackets,
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 5. Predict price using Scikit-Learn
app.post("/api/predict", async (req, res) => {
  try {
    const specs = req.body;
    const result = await runPythonPrediction(specs);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 6. Retrain models
app.post("/api/retrain", async (req, res) => {
  try {
    const output = await runPythonTraining();
    const metricsPath = path.join(__dirname, "models", "model_metrics.json");
    const metrics = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
    res.json({ success: true, output, metrics });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 7. Upload custom dataset
app.post("/api/upload", async (req, res) => {
  try {
    const { csvData } = req.body;
    if (!csvData || typeof csvData !== "string") {
      return res.status(400).json({ error: "Missing or invalid csvData in request body." });
    }
    const uploadPath = path.join(__dirname, "data", "laptop_prices.csv");
    fs.writeFileSync(uploadPath, csvData, "utf-8");
    const output = await runPythonTraining();
    const metricsPath = path.join(__dirname, "models", "model_metrics.json");
    const metrics = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
    res.json({ success: true, message: "Dataset uploaded and models successfully retrained!", metrics });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Start Express server and attach Vite middleware in development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // In production serve dist
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Laptop Price Analyzer server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
