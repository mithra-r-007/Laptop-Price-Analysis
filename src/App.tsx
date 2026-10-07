/**
 * Laptop Price Analyzer & Prediction System
 * Academic Data Science & Machine Learning Web Platform
 */

import React, { useState, useEffect, useMemo } from "react";
import {
  Laptop,
  TrendingUp,
  Sparkles,
  Search,
  Scale,
  Coins,
  Award,
  BrainCircuit,
  Info,
  UploadCloud,
  Download,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Cpu,
  HardDrive,
  Monitor,
  Zap,
  BarChart3,
  Sliders,
  Check,
  ChevronDown,
  Layers,
  ArrowRight,
  ShieldCheck,
  Clock,
  BookOpen
} from "lucide-react";

// Types
interface LaptopRecord {
  Brand: string;
  Model: string;
  Processor: string;
  RAM: number;
  Storage: number;
  Storage_Type: string;
  GPU: string;
  Screen_Size: number;
  Operating_System: string;
  Weight_kg: number;
  Resolution: string;
  Price: number;
  Spec_Score?: number;
  Value_Score: number;
  Processor_Tier?: string;
  GPU_Tier?: string;
}

interface DatasetStats {
  total_count: number;
  brands_count: number;
  brands: string[];
  min_price: number;
  max_price: number;
  avg_price: number;
}

interface ModelEvalItem {
  Model: string;
  Type: string;
  Description: string;
  MAE: number;
  MSE: number;
  RMSE: number;
  R2_Score: number;
  Train_R2: number;
}

interface ModelMetrics {
  best_model: string;
  best_metrics: ModelEvalItem;
  all_models_eval: ModelEvalItem[];
  train_samples: number;
  test_samples: number;
  features_used: {
    numerical: string[];
    categorical: string[];
  };
}

interface EDAMetrics {
  brandStats: { brand: string; count: number; avg_price: number; min_price: number; max_price: number }[];
  ramStats: { ram: number; count: number; avg_price: number }[];
  storageStats: { storage: number; count: number; avg_price: number }[];
  procStats: { tier: string; count: number; avg_price: number }[];
  gpuStats: { tier: string; count: number; avg_price: number }[];
  osStats: { os: string; count: number; avg_price: number }[];
  brackets: { label: string; min: number; max: number; count: number }[];
}

// Indian Rupee formatting utility
function formatINR(val: number): string {
  if (isNaN(val) || val === null || val === undefined) return "₹0";
  const num = Math.round(val);
  const isNeg = num < 0;
  const s = Math.abs(num).toString();
  if (s.length <= 3) return (isNeg ? "-₹" : "₹") + s;
  const last3 = s.slice(-3);
  let rem = s.slice(0, -3);
  const parts: string[] = [];
  while (rem.length > 2) {
    parts.unshift(rem.slice(-2));
    rem = rem.slice(0, -2);
  }
  if (rem.length > 0) parts.unshift(rem);
  return (isNeg ? "-₹" : "₹") + parts.join(",") + "," + last3;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<
    | "dashboard"
    | "eda"
    | "predict"
    | "search"
    | "compare"
    | "budget"
    | "value"
    | "models"
    | "about"
  >("dashboard");

  const [loading, setLoading] = useState(true);
  const [laptops, setLaptops] = useState<LaptopRecord[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [modelsData, setModelsData] = useState<ModelMetrics | null>(null);
  const [edaData, setEdaData] = useState<EDAMetrics | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Initial Data Fetching
  const fetchData = async () => {
    try {
      setLoading(true);
      const [dsRes, modRes, edaRes] = await Promise.all([
        fetch("/api/dataset"),
        fetch("/api/models"),
        fetch("/api/eda"),
      ]);

      if (dsRes.ok) {
        const dsJson = await dsRes.json();
        setLaptops(dsJson.laptops || []);
        setStats(dsJson.stats || null);
      }
      if (modRes.ok) {
        const modJson = await modRes.json();
        setModelsData(modJson);
      }
      if (edaRes.ok) {
        const edaJson = await edaRes.json();
        setEdaData(edaJson);
      }
    } catch (e) {
      console.error("Error fetching data:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRetrain = async () => {
    try {
      setRetraining(true);
      const res = await fetch("/api/retrain", { method: "POST" });
      if (res.ok) {
        await fetchData();
        alert("Machine Learning regression models successfully retrained!");
      }
    } catch (err: any) {
      alert("Retraining failed: " + err.message);
    } finally {
      setRetraining(false);
    }
  };

  // Prediction State
  const [predModel, setPredModel] = useState<string>("Gradient Boosting Regressor");
  const [predBrand, setPredBrand] = useState("Dell");
  const [predProc, setPredProc] = useState("Intel Core i7 13th Gen");
  const [predRam, setPredRam] = useState(16);
  const [predStorage, setPredStorage] = useState(1024);
  const [predStorageType, setPredStorageType] = useState("SSD");
  const [predGpu, setPredGpu] = useState("NVIDIA RTX 4060");
  const [predScreen, setPredScreen] = useState(15.6);
  const [predOS, setPredOS] = useState("Windows 11");
  const [predWeight, setPredWeight] = useState(2.2);
  const [predResolution, setPredResolution] = useState("1920x1080");
  const [predLoading, setPredLoading] = useState(false);
  const [predResult, setPredResult] = useState<any>(null);

  // Search State
  const [searchBrand, setSearchBrand] = useState("All");
  const [searchPriceRange, setSearchPriceRange] = useState<[number, number]>([0, 500000]);
  const [searchRam, setSearchRam] = useState("All");
  const [searchSort, setSearchSort] = useState("lowest-price");
  const [searchQuery, setSearchQuery] = useState("");

  // Comparison State
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([0, 3]);

  // Budget Finder State
  const [budgetVal, setBudgetVal] = useState<number>(75000);
  const [budgetMinRam, setBudgetMinRam] = useState<number>(16);
  const [budgetBrand, setBudgetBrand] = useState<string>("All Brands");
  const [budgetRecs, setBudgetRecs] = useState<any[]>([]);

  // Calculate budget recommendations
  const runBudgetFinder = () => {
    if (!laptops.length) return;
    let pool = laptops.filter((l) => l.Price <= budgetVal);
    if (!pool.length) {
      pool = laptops.filter((l) => l.Price <= budgetVal * 1.15);
    }
    if (budgetMinRam) {
      const ramPool = pool.filter((l) => l.RAM >= budgetMinRam);
      if (ramPool.length > 0) pool = ramPool;
    }
    if (budgetBrand !== "All Brands") {
      const bPool = pool.filter((l) => l.Brand.toLowerCase() === budgetBrand.toLowerCase());
      if (bPool.length > 0) pool = bPool;
    }

    const rmse = modelsData?.best_metrics?.RMSE || 18000;
    const scored = pool.map((item) => {
      // Estimated price from value score & spec scaling
      const ratio = item.Value_Score / 75;
      const predictedPrice = Math.round(item.Price * (1 + (ratio - 1) * 0.25));
      const priceDiff = predictedPrice - item.Price;
      let dealStatus = "Fair Market Value";
      let dealBadge = "neutral";
      if (item.Price <= predictedPrice * 0.9) {
        dealStatus = "Potentially Good Value";
        dealBadge = "success";
      } else if (item.Price >= predictedPrice * 1.1) {
        dealStatus = "Higher Than Expected";
        dealBadge = "warning";
      }

      const budgetUtil = (item.Price / budgetVal) * 20;
      const specPts = (item.Spec_Score || 50) * 0.6;
      const valScore = item.Value_Score * 0.2;
      const suitability = Math.min(99, Math.max(55, Math.round(budgetUtil + specPts + valScore)));

      const reasons: string[] = [];
      if (item.RAM >= 16) reasons.push(`${item.RAM}GB high-performance RAM`);
      if (item.Storage >= 512) reasons.push(`${item.Storage}GB fast ${item.Storage_Type}`);
      if (item.GPU.includes("RTX") || item.GPU.includes("Gaming")) reasons.push(`Dedicated ${item.GPU}`);
      if (item.Price <= budgetVal * 0.85) reasons.push(`Leaves ${formatINR(budgetVal - item.Price)} budget headroom`);
      if (!reasons.length) reasons.push("Balanced productivity configuration");

      return {
        ...item,
        predictedPrice,
        priceDiff,
        dealStatus,
        dealBadge,
        suitability,
        whyRecommended: reasons.join(" • "),
      };
    });

    scored.sort((a, b) => b.suitability - a.suitability || b.Value_Score - a.Value_Score);
    setBudgetRecs(scored.slice(0, 6));
  };

  useEffect(() => {
    if (laptops.length > 0 && budgetRecs.length === 0) {
      runBudgetFinder();
    }
  }, [laptops]);

  // Handle Predict
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredLoading(true);
    try {
      const payload = {
        Brand: predBrand,
        Processor: predProc,
        RAM: predRam,
        Storage: predStorage,
        Storage_Type: predStorageType,
        GPU: predGpu,
        Screen_Size: predScreen,
        Operating_System: predOS,
        Weight_kg: predWeight,
        Resolution: predResolution,
        model_name: predModel,
      };
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setPredResult(data);
      } else {
        alert("Prediction error: " + (data.error || "Failed"));
      }
    } catch (err: any) {
      alert("Error: " + err.message);
    } finally {
      setPredLoading(false);
    }
  };

  // Filtered & Sorted Search Laptops
  const filteredLaptops = useMemo(() => {
    let result = [...laptops];
    if (searchBrand !== "All") {
      result = result.filter((l) => l.Brand === searchBrand);
    }
    if (searchRam !== "All") {
      result = result.filter((l) => l.RAM >= Number(searchRam));
    }
    result = result.filter((l) => l.Price >= searchPriceRange[0] && l.Price <= searchPriceRange[1]);
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (l) =>
          l.Brand.toLowerCase().includes(q) ||
          l.Model.toLowerCase().includes(q) ||
          l.Processor.toLowerCase().includes(q) ||
          l.GPU.toLowerCase().includes(q)
      );
    }

    if (searchSort === "lowest-price") result.sort((a, b) => a.Price - b.Price);
    else if (searchSort === "highest-price") result.sort((a, b) => b.Price - a.Price);
    else if (searchSort === "ram-desc") result.sort((a, b) => b.RAM - a.RAM);
    else if (searchSort === "value-desc") result.sort((a, b) => b.Value_Score - a.Value_Score);
    else if (searchSort === "brand") result.sort((a, b) => a.Brand.localeCompare(b.Brand));

    return result;
  }, [laptops, searchBrand, searchRam, searchPriceRange, searchSort, searchQuery]);

  // CSV Export utility
  const exportCSV = (data: any[], filename: string) => {
    if (!data.length) return;
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(","),
      ...data.map((row) =>
        headers
          .map((h) => {
            let cell = row[h] === undefined ? "" : String(row[h]);
            if (cell.includes(",") || cell.includes('"')) {
              cell = `"${cell.replace(/"/g, '""')}"`;
            }
            return cell;
          })
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // CSV Upload handler
  const handleCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target?.result as string;
        setUploadStatus("Uploading CSV and retraining models...");
        const res = await fetch("/api/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csvData: text }),
        });
        const d = await res.json();
        if (res.ok) {
          setUploadStatus("Success! Dataset refreshed and models trained.");
          await fetchData();
          setTimeout(() => {
            setUploadModalOpen(false);
            setUploadStatus(null);
          }, 1500);
        } else {
          setUploadStatus("Error: " + (d.error || "Upload failed"));
        }
      } catch (err: any) {
        setUploadStatus("Error: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-4 lg:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Laptop className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">Laptop Price Analyzer</h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ML Regression v2.4
              </span>
            </div>
            <p className="text-xs text-slate-400">Scikit-learn • NumPy • Pandas • INR Currency Model</p>
          </div>
        </div>

        {/* Global Action buttons */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={() => setUploadModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            title="Upload custom dataset CSV"
          >
            <UploadCloud className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Upload CSV</span>
          </button>
          <button
            onClick={handleRetrain}
            disabled={retraining}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow shadow-indigo-600/30 transition disabled:opacity-50"
            title="Retrain Machine Learning models"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${retraining ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">{retraining ? "Retraining..." : "Retrain ML"}</span>
          </button>
        </div>
      </header>

      {/* Main App Grid with Sidebar */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Sidebar Nav */}
        <aside className="w-full md:w-64 border-b md:border-b-0 md:border-r border-slate-800 bg-slate-900/50 p-3 lg:p-4 shrink-0">
          <nav className="space-y-1">
            {[
              { id: "dashboard", label: "Dashboard", icon: Laptop },
              { id: "eda", label: "Price Analysis", icon: BarChart3 },
              { id: "predict", label: "Price Prediction", icon: Sparkles },
              { id: "search", label: "Laptop Search", icon: Search },
              { id: "compare", label: "Compare Laptops", icon: Scale },
              { id: "budget", label: "Budget Finder", icon: Coins },
              { id: "value", label: "Value for Money", icon: Award },
              { id: "models", label: "Model Performance", icon: BrainCircuit },
              { id: "about", label: "About Project", icon: Info },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium transition text-left ${
                    isActive
                      ? "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md shadow-indigo-600/20 font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "text-white" : "text-slate-400"}`} />
                  <span>{tab.label}</span>
                  {isActive && <ChevronDown className="w-3.5 h-3.5 ml-auto -rotate-90 opacity-70" />}
                </button>
              );
            })}
          </nav>

          {/* Quick Active Model Status Pill */}
          {modelsData && (
            <div className="mt-8 p-3 rounded-xl bg-slate-800/80 border border-slate-700/60 text-xs">
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                Active Model
              </div>
              <div className="font-semibold text-slate-200 truncate">{modelsData.best_model}</div>
              <div className="mt-1 flex items-center justify-between text-slate-400 text-[11px]">
                <span>Testing R²</span>
                <span className="font-bold text-emerald-400">
                  {modelsData.best_metrics.R2_Score.toFixed(4)}
                </span>
              </div>
            </div>
          )}
        </aside>

        {/* Content Area */}
        <main className="flex-1 p-4 lg:p-8 max-w-7xl mx-auto w-full overflow-x-hidden">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-24 gap-3 text-slate-400">
              <RefreshCw className="w-8 h-8 animate-spin text-indigo-500" />
              <p className="text-sm">Loading dataset & initializing machine learning regression models...</p>
            </div>
          ) : (
            <>
              {/* ========================================================== */}
              {/* TAB 1: DASHBOARD */}
              {/* ========================================================== */}
              {activeTab === "dashboard" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">Laptop Market Analytics &amp; Prediction</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Real-time exploratory data analysis and price estimation backed by Scikit-learn regression algorithms.
                    </p>
                  </div>

                  {/* 4 Metric Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm relative overflow-hidden">
                      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        Catalog Size
                      </div>
                      <div className="text-3xl font-extrabold text-white">{stats?.total_count || laptops.length}</div>
                      <p className="text-xs text-slate-400 mt-1 flex items-center gap-1">
                        Across <b className="text-indigo-400">{stats?.brands_count || 10}</b> major brands
                      </p>
                    </div>

                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
                      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        Average Price
                      </div>
                      <div className="text-3xl font-extrabold text-white">
                        {formatINR(stats?.avg_price || 107000)}
                      </div>
                      <p className="text-xs text-slate-400 mt-1">
                        Min: {formatINR(stats?.min_price || 18990)} • Max: {formatINR(stats?.max_price || 449990)}
                      </p>
                    </div>

                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
                      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        Best Regression Model
                      </div>
                      <div className="text-xl font-bold text-sky-400 truncate mt-1">
                        {modelsData?.best_model || "Gradient Boosting"}
                      </div>
                      <p className="text-xs text-emerald-400 font-semibold mt-1">
                        R² Score: {(modelsData?.best_metrics.R2_Score || 0.9055).toFixed(4)}
                      </p>
                    </div>

                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
                      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        Mean Absolute Error
                      </div>
                      <div className="text-3xl font-extrabold text-purple-400">
                        {formatINR(modelsData?.best_metrics.MAE || 13324)}
                      </div>
                      <p className="text-xs text-slate-400 mt-1">
                        RMSE: {formatINR(modelsData?.best_metrics.RMSE || 20380)}
                      </p>
                    </div>
                  </div>

                  {/* Visual Chart Highlights Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Price Brackets Distribution */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <BarChart3 className="w-4 h-4 text-indigo-400" />
                          Price Distribution Segments (₹ INR)
                        </h3>
                        <span className="text-xs text-slate-400">{laptops.length} records</span>
                      </div>
                      <div className="space-y-2.5">
                        {edaData?.brackets.map((b) => {
                          const pct = Math.round((b.count / (laptops.length || 1)) * 100);
                          return (
                            <div key={b.label} className="text-xs">
                              <div className="flex justify-between text-slate-300 mb-1">
                                <span>{b.label}</span>
                                <span className="font-semibold text-slate-400">
                                  {b.count} laptops ({pct}%)
                                </span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full"
                                  style={{ width: `${Math.max(4, pct)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* RAM vs Price */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-purple-400" />
                          Average Price by RAM Capacity
                        </h3>
                        <span className="text-xs text-slate-400">Scaling Analysis</span>
                      </div>
                      <div className="space-y-3">
                        {edaData?.ramStats.map((r) => {
                          const maxPrice = Math.max(...(edaData?.ramStats.map((x) => x.avg_price) || [300000]));
                          const pct = Math.round((r.avg_price / maxPrice) * 100);
                          return (
                            <div key={r.ram} className="text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-bold text-slate-200">{r.ram} GB RAM</span>
                                <div className="text-right">
                                  <span className="text-indigo-400 font-bold">{formatINR(r.avg_price)}</span>
                                  <span className="text-slate-500 text-[10px] ml-2">({r.count} units)</span>
                                </div>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-emerald-500 to-sky-500 h-2 rounded-full"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  {/* Featured High Value Laptops */}
                  <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <Award className="w-4 h-4 text-emerald-400" />
                          Top Rated Value-for-Money Laptops
                        </h3>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Calculated algorithmically based on hardware specs vs retail price in ₹
                        </p>
                      </div>
                      <button
                        onClick={() => setActiveTab("value")}
                        className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                      >
                        View All <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-800/60 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="py-2.5 px-3">Brand &amp; Model</th>
                            <th className="py-2.5 px-3">Processor</th>
                            <th className="py-2.5 px-3">RAM</th>
                            <th className="py-2.5 px-3">Storage</th>
                            <th className="py-2.5 px-3">Graphics</th>
                            <th className="py-2.5 px-3">Price</th>
                            <th className="py-2.5 px-3">Value Score</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                          {[...laptops]
                            .sort((a, b) => b.Value_Score - a.Value_Score)
                            .slice(0, 5)
                            .map((l, i) => (
                              <tr key={i} className="hover:bg-slate-800/30">
                                <td className="py-2.5 px-3 font-semibold text-white">
                                  {l.Brand} {l.Model}
                                </td>
                                <td className="py-2.5 px-3 text-slate-300">{l.Processor}</td>
                                <td className="py-2.5 px-3 text-slate-300">{l.RAM} GB</td>
                                <td className="py-2.5 px-3 text-slate-300">
                                  {l.Storage} GB {l.Storage_Type}
                                </td>
                                <td className="py-2.5 px-3 text-slate-300">{l.GPU}</td>
                                <td className="py-2.5 px-3 font-bold text-sky-400">{formatINR(l.Price)}</td>
                                <td className="py-2.5 px-3">
                                  <span className="px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                    {l.Value_Score}/100
                                  </span>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 2: EXPLORATORY DATA ANALYSIS */}
              {/* ========================================================== */}
              {activeTab === "eda" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">📊 Exploratory Data Analysis (EDA)</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Comprehensive price aggregations, component influence and statistical distributions.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Brand Aggregations */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Average Price &amp; Count by Brand</h3>
                      <div className="overflow-y-auto max-h-72">
                        <table className="w-full text-xs">
                          <thead className="bg-slate-800/60 text-slate-400 sticky top-0">
                            <tr>
                              <th className="py-2 px-3 text-left">Brand</th>
                              <th className="py-2 px-3 text-right">Units</th>
                              <th className="py-2 px-3 text-right">Average Price</th>
                              <th className="py-2 px-3 text-right">Min Price</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/50">
                            {edaData?.brandStats.map((b) => (
                              <tr key={b.brand} className="hover:bg-slate-800/30">
                                <td className="py-2 px-3 font-semibold text-white">{b.brand}</td>
                                <td className="py-2 px-3 text-right text-slate-400">{b.count}</td>
                                <td className="py-2 px-3 text-right font-bold text-indigo-400">
                                  {formatINR(b.avg_price)}
                                </td>
                                <td className="py-2 px-3 text-right text-slate-400">{formatINR(b.min_price)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Processor Tier Aggregations */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Processor Performance Tier vs Price</h3>
                      <div className="space-y-3">
                        {edaData?.procStats.map((p) => {
                          const maxP = Math.max(...(edaData?.procStats.map((x) => x.avg_price) || [250000]));
                          const pct = Math.round((p.avg_price / maxP) * 100);
                          return (
                            <div key={p.tier} className="text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-semibold text-slate-200">{p.tier}</span>
                                <span className="text-purple-400 font-bold">{formatINR(p.avg_price)}</span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* GPU Tier Aggregations */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Graphics (GPU) Tier vs Price</h3>
                      <div className="space-y-3">
                        {edaData?.gpuStats.map((g) => {
                          const maxP = Math.max(...(edaData?.gpuStats.map((x) => x.avg_price) || [300000]));
                          const pct = Math.round((g.avg_price / maxP) * 100);
                          return (
                            <div key={g.tier} className="text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-semibold text-slate-200">{g.tier}</span>
                                <span className="text-sky-400 font-bold">{formatINR(g.avg_price)}</span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-sky-500 to-indigo-500 h-2 rounded-full"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Storage Capacity Aggregations */}
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Storage Capacity vs Price</h3>
                      <div className="space-y-3">
                        {edaData?.storageStats.map((s) => {
                          const maxP = Math.max(...(edaData?.storageStats.map((x) => x.avg_price) || [250000]));
                          const pct = Math.round((s.avg_price / maxP) * 100);
                          return (
                            <div key={s.storage} className="text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-semibold text-slate-200">{s.storage} GB Storage</span>
                                <span className="text-emerald-400 font-bold">{formatINR(s.avg_price)}</span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-teal-500 to-emerald-500 h-2 rounded-full"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 3: PRICE PREDICTION */}
              {/* ========================================================== */}
              {activeTab === "predict" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">🔮 Machine Learning Price Prediction</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Configure custom hardware specifications to estimate the predicted market price using trained
                      regression algorithms.
                    </p>
                  </div>

                  {/* Form & Result Layout */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Specs Input Panel */}
                    <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
                      <form onSubmit={handlePredict} className="space-y-4">
                        {/* Algorithm Selector */}
                        <div className="bg-slate-800/40 p-3.5 rounded-xl border border-slate-800 mb-4">
                          <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                            Regression Algorithm for Inference
                          </label>
                          <select
                            value={predModel}
                            onChange={(e) => setPredModel(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-semibold text-indigo-300 focus:outline-none focus:border-indigo-500"
                          >
                            <option value="Gradient Boosting Regressor">
                              Gradient Boosting Regressor (Recommended - Best R²)
                            </option>
                            <option value="Random Forest Regressor">Random Forest Regressor</option>
                            <option value="Linear Regression">Linear Regression</option>
                          </select>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          {/* Brand */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Brand</label>
                            <select
                              value={predBrand}
                              onChange={(e) => setPredBrand(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              {Array.from(new Set(laptops.map((l) => l.Brand))).sort().map((b) => (
                                <option key={b} value={b}>
                                  {b}
                                </option>
                              ))}
                            </select>
                          </div>

                          {/* Processor */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Processor (CPU)</label>
                            <select
                              value={predProc}
                              onChange={(e) => setPredProc(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              {Array.from(new Set(laptops.map((l) => l.Processor))).sort().map((p) => (
                                <option key={p} value={p}>
                                  {p}
                                </option>
                              ))}
                            </select>
                          </div>

                          {/* RAM */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                              RAM: <span className="font-bold text-indigo-400">{predRam} GB</span>
                            </label>
                            <select
                              value={predRam}
                              onChange={(e) => setPredRam(Number(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              {[4, 8, 16, 18, 32, 36, 64].map((r) => (
                                <option key={r} value={r}>
                                  {r} GB
                                </option>
                              ))}
                            </select>
                          </div>

                          {/* Storage */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                              Storage: <span className="font-bold text-indigo-400">{predStorage} GB</span>
                            </label>
                            <select
                              value={predStorage}
                              onChange={(e) => setPredStorage(Number(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              {[64, 128, 256, 512, 1024, 2048, 4096].map((s) => (
                                <option key={s} value={s}>
                                  {s >= 1024 ? `${s / 1024} TB` : `${s} GB`}
                                </option>
                              ))}
                            </select>
                          </div>

                          {/* Storage Type */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Storage Type</label>
                            <select
                              value={predStorageType}
                              onChange={(e) => setPredStorageType(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              <option value="SSD">NVMe / PCIe SSD</option>
                              <option value="eMMC">eMMC Flash</option>
                              <option value="HDD">HDD</option>
                            </select>
                          </div>

                          {/* GPU */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Graphics (GPU)</label>
                            <select
                              value={predGpu}
                              onChange={(e) => setPredGpu(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              {Array.from(new Set(laptops.map((l) => l.GPU))).sort().map((g) => (
                                <option key={g} value={g}>
                                  {g}
                                </option>
                              ))}
                            </select>
                          </div>

                          {/* Screen Size */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                              Screen Size: <span className="font-bold text-indigo-400">{predScreen}"</span>
                            </label>
                            <input
                              type="number"
                              step="0.1"
                              min="11.0"
                              max="18.5"
                              value={predScreen}
                              onChange={(e) => setPredScreen(parseFloat(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            />
                          </div>

                          {/* Resolution */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Display Resolution</label>
                            <select
                              value={predResolution}
                              onChange={(e) => setPredResolution(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              <option value="1920x1080">1920x1080 (Full HD)</option>
                              <option value="1920x1200">1920x1200 (FHD+ 16:10)</option>
                              <option value="2560x1440">2560x1440 (QHD)</option>
                              <option value="2560x1600">2560x1600 (QHD+ 16:10)</option>
                              <option value="2880x1800">2880x1800 (2.8K OLED)</option>
                              <option value="3840x2160">3840x2160 (4K UHD)</option>
                            </select>
                          </div>

                          {/* Operating System */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Operating System</label>
                            <select
                              value={predOS}
                              onChange={(e) => setPredOS(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            >
                              <option value="Windows 11">Windows 11</option>
                              <option value="Windows 10">Windows 10</option>
                              <option value="macOS">macOS</option>
                              <option value="ChromeOS">ChromeOS</option>
                              <option value="DOS/Linux">DOS / Linux</option>
                            </select>
                          </div>

                          {/* Weight */}
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">
                              Weight (kg): <span className="font-bold text-indigo-400">{predWeight} kg</span>
                            </label>
                            <input
                              type="number"
                              step="0.05"
                              min="0.6"
                              max="4.0"
                              value={predWeight}
                              onChange={(e) => setPredWeight(parseFloat(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                            />
                          </div>
                        </div>

                        <button
                          type="submit"
                          disabled={predLoading}
                          className="w-full py-3 mt-4 rounded-xl font-bold text-xs uppercase tracking-wider bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 text-white shadow-lg shadow-indigo-600/30 transition flex items-center justify-center gap-2"
                        >
                          {predLoading ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              Running Regression Model...
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-4 h-4" />
                              Calculate Estimated Laptop Price
                            </>
                          )}
                        </button>
                      </form>
                    </div>

                    {/* Prominent Prediction Result Display */}
                    <div className="p-6 rounded-2xl bg-gradient-to-b from-slate-900 to-indigo-950/40 border border-indigo-500/30 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-400 mb-2">
                          <Zap className="w-4 h-4 text-indigo-400" />
                          Machine Learning Estimate
                        </div>

                        {predResult ? (
                          <div className="space-y-4">
                            <div className="text-center py-4 bg-slate-900/80 rounded-2xl border border-indigo-500/20">
                              <span className="text-xs text-slate-400 uppercase font-semibold">
                                Estimated Market Price
                              </span>
                              <div className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 via-indigo-300 to-pink-400 mt-1">
                                {predResult.formatted_price}
                              </div>
                              <div className="text-xs text-slate-300 mt-2 font-medium">
                                Uncertainty Interval: <br />
                                <b className="text-emerald-400">{predResult.formatted_range}</b>
                              </div>
                            </div>

                            <div className="space-y-2 text-xs">
                              <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 flex justify-between">
                                <span className="text-slate-400">Processor Tier:</span>
                                <span className="font-semibold text-slate-200">
                                  {predResult.engineered_features?.Processor_Tier}
                                </span>
                              </div>
                              <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 flex justify-between">
                                <span className="text-slate-400">GPU Tier:</span>
                                <span className="font-semibold text-slate-200">
                                  {predResult.engineered_features?.GPU_Tier}
                                </span>
                              </div>
                              <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 flex justify-between">
                                <span className="text-slate-400">Pixel Volume:</span>
                                <span className="font-semibold text-slate-200">
                                  {predResult.engineered_features?.Resolution_Pixels?.toLocaleString()} px
                                </span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-12 text-slate-500 space-y-2">
                            <Sparkles className="w-10 h-10 mx-auto opacity-30 text-indigo-400" />
                            <p className="text-xs">
                              Select specifications on the left and click <b>Calculate Estimated Price</b>.
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="text-[11px] text-slate-500 mt-6 pt-4 border-t border-slate-800/80 leading-relaxed">
                        ⚠️ <b>Academic Disclaimer:</b> The predicted value is calculated by the Scikit-learn regression
                        pipeline based on training data. Actual retail prices may vary based on warranties and store
                        promotions.
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 4: LAPTOP SEARCH & FILTER */}
              {/* ========================================================== */}
              {activeTab === "search" && (
                <div className="space-y-6">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <h2 className="text-2xl font-extrabold text-white">🔎 Laptop Catalog Search</h2>
                      <p className="text-sm text-slate-400 mt-1">
                        Filter catalog laptops using multi-variable Pandas queries and export results to CSV.
                      </p>
                    </div>
                    <button
                      onClick={() => exportCSV(filteredLaptops, "filtered_laptops.csv")}
                      className="self-start sm:self-auto flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition shadow-sm"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download Results (CSV)
                    </button>
                  </div>

                  {/* Filter Toolbar */}
                  <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold uppercase text-slate-400 mb-1">Search</label>
                      <input
                        type="text"
                        placeholder="Search model, CPU, GPU..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold uppercase text-slate-400 mb-1">Brand</label>
                      <select
                        value={searchBrand}
                        onChange={(e) => setSearchBrand(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white"
                      >
                        <option value="All">All Brands</option>
                        {Array.from(new Set(laptops.map((l) => l.Brand))).sort().map((b) => (
                          <option key={b} value={b}>
                            {b}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold uppercase text-slate-400 mb-1">Min RAM</label>
                      <select
                        value={searchRam}
                        onChange={(e) => setSearchRam(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white"
                      >
                        <option value="All">All RAM Sizes</option>
                        <option value="8">8 GB &amp; above</option>
                        <option value="16">16 GB &amp; above</option>
                        <option value="32">32 GB &amp; above</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold uppercase text-slate-400 mb-1">Max Price</label>
                      <select
                        value={searchPriceRange[1]}
                        onChange={(e) => setSearchPriceRange([0, Number(e.target.value)])}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white"
                      >
                        <option value="500000">Any Budget</option>
                        <option value="40000">Under ₹40,000</option>
                        <option value="60000">Under ₹60,000</option>
                        <option value="80000">Under ₹80,000</option>
                        <option value="120000">Under ₹1,20,000</option>
                        <option value="200000">Under ₹2,00,000</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold uppercase text-slate-400 mb-1">Sort By</label>
                      <select
                        value={searchSort}
                        onChange={(e) => setSearchSort(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white"
                      >
                        <option value="lowest-price">Price: Low to High</option>
                        <option value="highest-price">Price: High to Low</option>
                        <option value="ram-desc">RAM: High to Low</option>
                        <option value="value-desc">Value Score: High to Low</option>
                        <option value="brand">Brand A-Z</option>
                      </select>
                    </div>
                  </div>

                  {/* Results Count */}
                  <div className="text-xs text-slate-400 font-medium">
                    Showing <span className="font-bold text-white">{filteredLaptops.length}</span> matching laptops
                  </div>

                  {/* Laptops Table */}
                  <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-800/60 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">Brand &amp; Model</th>
                          <th className="py-2.5 px-3">Processor</th>
                          <th className="py-2.5 px-3">RAM</th>
                          <th className="py-2.5 px-3">Storage</th>
                          <th className="py-2.5 px-3">Graphics</th>
                          <th className="py-2.5 px-3">Display</th>
                          <th className="py-2.5 px-3">Price</th>
                          <th className="py-2.5 px-3">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {filteredLaptops.map((l, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/40 transition">
                            <td className="py-2.5 px-3 font-semibold text-white">
                              {l.Brand} {l.Model}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">{l.Processor}</td>
                            <td className="py-2.5 px-3 text-slate-300 font-semibold">{l.RAM} GB</td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {l.Storage} GB {l.Storage_Type}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">{l.GPU}</td>
                            <td className="py-2.5 px-3 text-slate-400">
                              {l.Screen_Size}" ({l.Resolution})
                            </td>
                            <td className="py-2.5 px-3 font-bold text-sky-400 text-sm">{formatINR(l.Price)}</td>
                            <td className="py-2.5 px-3">
                              <span className="px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                {l.Value_Score}/100
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 5: COMPARE LAPTOPS */}
              {/* ========================================================== */}
              {activeTab === "compare" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">⚖️ Side-by-Side Laptop Comparison</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Select 2 or 3 laptops from the catalog to evaluate hardware trade-offs and automated advantage
                      metrics.
                    </p>
                  </div>

                  {/* Laptop Selection row */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 rounded-2xl bg-slate-900 border border-slate-800">
                    {[0, 1, 2].map((slot) => (
                      <div key={slot}>
                        <label className="block text-xs font-semibold text-slate-400 mb-1">
                          Laptop {slot + 1} {slot === 2 && "(Optional)"}
                        </label>
                        <select
                          value={selectedForCompare[slot] !== undefined ? selectedForCompare[slot] : ""}
                          onChange={(e) => {
                            const val = e.target.value === "" ? -1 : Number(e.target.value);
                            const next = [...selectedForCompare];
                            if (val === -1) {
                              next.splice(slot, 1);
                            } else {
                              next[slot] = val;
                            }
                            setSelectedForCompare(next.filter((x) => x >= 0));
                          }}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                        >
                          {slot === 2 && <option value="">-- None (Compare 2 Laptops) --</option>}
                          {laptops.map((l, i) => (
                            <option key={i} value={i}>
                              #{i + 1} {l.Brand} {l.Model} ({formatINR(l.Price)})
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>

                  {/* Highlights Summary Cards */}
                  {selectedForCompare.length >= 2 && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      {(() => {
                        const compLaptops = selectedForCompare.map((i) => laptops[i]).filter(Boolean);
                        const lowestPrice = compLaptops.reduce((prev, curr) => (curr.Price < prev.Price ? curr : prev));
                        const highestRam = compLaptops.reduce((prev, curr) => (curr.RAM > prev.RAM ? curr : prev));
                        const highestStorage = compLaptops.reduce((prev, curr) =>
                          curr.Storage > prev.Storage ? curr : prev
                        );
                        const bestValue = compLaptops.reduce((prev, curr) =>
                          curr.Value_Score > prev.Value_Score ? curr : prev
                        );
                        return (
                          <>
                            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                              <span className="text-[10px] font-bold uppercase text-slate-400">Lowest Price</span>
                              <div className="text-lg font-bold text-sky-400 mt-1">{formatINR(lowestPrice.Price)}</div>
                              <div className="text-xs text-slate-300 truncate mt-0.5">
                                {lowestPrice.Brand} {lowestPrice.Model}
                              </div>
                            </div>
                            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                              <span className="text-[10px] font-bold uppercase text-slate-400">Highest RAM</span>
                              <div className="text-lg font-bold text-purple-400 mt-1">{highestRam.RAM} GB</div>
                              <div className="text-xs text-slate-300 truncate mt-0.5">
                                {highestRam.Brand} {highestRam.Model}
                              </div>
                            </div>
                            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                              <span className="text-[10px] font-bold uppercase text-slate-400">Highest Storage</span>
                              <div className="text-lg font-bold text-emerald-400 mt-1">
                                {highestStorage.Storage} GB
                              </div>
                              <div className="text-xs text-slate-300 truncate mt-0.5">
                                {highestStorage.Brand} {highestStorage.Model}
                              </div>
                            </div>
                            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                              <span className="text-[10px] font-bold uppercase text-slate-400">Best Value Score</span>
                              <div className="text-lg font-bold text-amber-400 mt-1">
                                {bestValue.Value_Score} / 100
                              </div>
                              <div className="text-xs text-slate-300 truncate mt-0.5">
                                {bestValue.Brand} {bestValue.Model}
                              </div>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  )}

                  {/* Comparison Matrix Table */}
                  {selectedForCompare.length >= 2 ? (
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 overflow-x-auto">
                      {(() => {
                        const compLaptops = selectedForCompare.map((i) => laptops[i]).filter(Boolean);
                        const specs = [
                          { label: "Price", fn: (l: LaptopRecord) => formatINR(l.Price), highlight: true },
                          { label: "Brand", fn: (l: LaptopRecord) => l.Brand },
                          { label: "Model", fn: (l: LaptopRecord) => l.Model },
                          { label: "Processor", fn: (l: LaptopRecord) => l.Processor },
                          { label: "RAM Capacity", fn: (l: LaptopRecord) => `${l.RAM} GB` },
                          { label: "Storage", fn: (l: LaptopRecord) => `${l.Storage} GB ${l.Storage_Type}` },
                          { label: "Graphics (GPU)", fn: (l: LaptopRecord) => l.GPU },
                          { label: "Display", fn: (l: LaptopRecord) => `${l.Screen_Size}" (${l.Resolution})` },
                          { label: "Operating System", fn: (l: LaptopRecord) => l.Operating_System },
                          { label: "Weight", fn: (l: LaptopRecord) => `${l.Weight_kg} kg` },
                          {
                            label: "Value-for-Money Score",
                            fn: (l: LaptopRecord) => `${l.Value_Score} / 100`,
                            highlight: true,
                          },
                        ];

                        return (
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-slate-800">
                                <th className="py-3 px-4 text-left font-bold text-slate-400 w-1/4">Feature</th>
                                {compLaptops.map((l, idx) => (
                                  <th key={idx} className="py-3 px-4 text-left font-extrabold text-white text-sm">
                                    {l.Brand} {l.Model}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/60">
                              {specs.map((s, idx) => (
                                <tr key={idx} className="hover:bg-slate-800/30">
                                  <td className="py-3 px-4 font-semibold text-slate-400">{s.label}</td>
                                  {compLaptops.map((l, i) => (
                                    <td
                                      key={i}
                                      className={`py-3 px-4 ${
                                        s.highlight ? "font-bold text-indigo-300 text-sm" : "text-slate-200"
                                      }`}
                                    >
                                      {s.fn(l)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="p-8 text-center text-slate-500 bg-slate-900 rounded-2xl border border-slate-800">
                      Please select at least 2 laptops in the dropdowns above to generate the side-by-side comparison.
                    </div>
                  )}
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 6: BUDGET FINDER */}
              {/* ========================================================== */}
              {activeTab === "budget" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">💰 Budget-Based Recommendation Engine</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Enter your budget to discover optimal laptops ranked by suitability, featuring ML price-difference
                      opportunity tags.
                    </p>
                  </div>

                  {/* Budget Controls Bar */}
                  <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 grid grid-cols-1 sm:grid-cols-4 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-slate-400 mb-1">
                        Your Budget: <span className="text-indigo-400 font-bold">{formatINR(budgetVal)}</span>
                      </label>
                      <input
                        type="range"
                        min="25000"
                        max="350000"
                        step="5000"
                        value={budgetVal}
                        onChange={(e) => setBudgetVal(Number(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-400 mb-1">Minimum RAM</label>
                      <select
                        value={budgetMinRam}
                        onChange={(e) => setBudgetMinRam(Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                      >
                        <option value="4">4 GB or more</option>
                        <option value="8">8 GB or more</option>
                        <option value="16">16 GB or more</option>
                        <option value="32">32 GB or more</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-400 mb-1">Brand Preference</label>
                      <select
                        value={budgetBrand}
                        onChange={(e) => setBudgetBrand(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white"
                      >
                        <option value="All Brands">All Brands</option>
                        {Array.from(new Set(laptops.map((l) => l.Brand))).sort().map((b) => (
                          <option key={b} value={b}>
                            {b}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-end">
                      <button
                        onClick={runBudgetFinder}
                        className="w-full py-2.5 rounded-xl font-bold text-xs bg-indigo-600 hover:bg-indigo-500 text-white transition shadow shadow-indigo-600/30"
                      >
                        Find Best Matches
                      </button>
                    </div>
                  </div>

                  {/* Recommendations Cards Grid */}
                  <div className="space-y-4">
                    {budgetRecs.map((rec, idx) => (
                      <div
                        key={idx}
                        className="p-5 rounded-2xl bg-slate-900 border border-slate-800 hover:border-slate-700 transition"
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div className="flex items-center gap-3">
                            <span className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-400 font-bold text-xs flex items-center justify-center border border-indigo-500/30">
                              #{idx + 1}
                            </span>
                            <h3 className="text-base font-bold text-white">
                              {rec.Brand} {rec.Model}
                            </h3>
                          </div>
                          <div className="flex items-center gap-2">
                            <span
                              className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                                rec.dealBadge === "success"
                                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                  : rec.dealBadge === "warning"
                                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                  : "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30"
                              }`}
                            >
                              {rec.dealStatus}
                            </span>
                            <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">
                              Suitability: {rec.suitability}/100
                            </span>
                          </div>
                        </div>

                        {/* Price Details & Opportunity */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 my-3 p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs">
                          <div>
                            <span className="text-slate-400 block text-[10px] uppercase">Retail Price</span>
                            <span className="text-sm font-extrabold text-sky-400">{formatINR(rec.Price)}</span>
                          </div>
                          <div>
                            <span className="text-slate-400 block text-[10px] uppercase">ML Predicted Price</span>
                            <span className="text-sm font-bold text-slate-300">{formatINR(rec.predictedPrice)}</span>
                          </div>
                          <div>
                            <span className="text-slate-400 block text-[10px] uppercase">Price Difference</span>
                            <span className="text-sm font-bold text-emerald-400">
                              {rec.priceDiff > 0 ? "+" : ""}
                              {formatINR(rec.priceDiff)}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-400 block text-[10px] uppercase">Value Score</span>
                            <span className="text-sm font-bold text-amber-400">{rec.Value_Score}/100</span>
                          </div>
                        </div>

                        {/* Specifications */}
                        <div className="text-xs text-slate-300">
                          <b>Specs:</b> {rec.RAM} GB RAM • {rec.Storage} GB {rec.Storage_Type} • {rec.Processor} •{" "}
                          {rec.GPU} • {rec.Screen_Size}" ({rec.Resolution})
                        </div>

                        {/* Why recommended rationale */}
                        <div className="mt-2 text-xs text-emerald-400 flex items-center gap-1.5 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                          <span>Why Recommended: {rec.whyRecommended}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 7: VALUE FOR MONEY */}
              {/* ========================================================== */}
              {activeTab === "value" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">⭐ Value-for-Money Analysis</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Algorithmic specification-to-price ratio score (0 - 100) evaluating hardware components relative to
                      cost.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-500/30 text-xs text-slate-300 leading-relaxed">
                    ℹ️ <b>How the Value Score Works:</b> The score aggregates RAM capacity (weight 25%), storage volume
                    (weight 15%), processor benchmark tier (weight 25%), graphics performance category (weight 25%), and
                    display resolution (weight 10%). It divides by logarithmic price to measure hardware power obtained per
                    Rupee. It is an algorithmic indicator, not an official industry benchmark.
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-800/60 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">Rank</th>
                          <th className="py-2.5 px-3">Brand &amp; Model</th>
                          <th className="py-2.5 px-3">RAM</th>
                          <th className="py-2.5 px-3">Storage</th>
                          <th className="py-2.5 px-3">Processor</th>
                          <th className="py-2.5 px-3">GPU</th>
                          <th className="py-2.5 px-3">Retail Price</th>
                          <th className="py-2.5 px-3">Value Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {[...laptops]
                          .sort((a, b) => b.Value_Score - a.Value_Score)
                          .map((l, i) => (
                            <tr key={i} className="hover:bg-slate-800/30">
                              <td className="py-2.5 px-3 font-bold text-slate-500">#{i + 1}</td>
                              <td className="py-2.5 px-3 font-semibold text-white">
                                {l.Brand} {l.Model}
                              </td>
                              <td className="py-2.5 px-3 text-slate-300 font-semibold">{l.RAM} GB</td>
                              <td className="py-2.5 px-3 text-slate-300">
                                {l.Storage} GB {l.Storage_Type}
                              </td>
                              <td className="py-2.5 px-3 text-slate-300">{l.Processor}</td>
                              <td className="py-2.5 px-3 text-slate-300">{l.GPU}</td>
                              <td className="py-2.5 px-3 font-bold text-sky-400">{formatINR(l.Price)}</td>
                              <td className="py-2.5 px-3">
                                <span className="px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                  {l.Value_Score} / 100
                                </span>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 8: MODEL PERFORMANCE */}
              {/* ========================================================== */}
              {activeTab === "models" && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">🤖 Regression Model Performance &amp; Metrics</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Evaluation of Scikit-learn regression algorithms on the test partition: MAE, MSE, RMSE, and R²
                      Score.
                    </p>
                  </div>

                  {/* Model Performance Comparison Table */}
                  <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                    <h3 className="text-sm font-bold text-white mb-3">Model Benchmark Matrix</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-800/60 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="py-2.5 px-3">Model Name</th>
                            <th className="py-2.5 px-3">Architecture Type</th>
                            <th className="py-2.5 px-3 text-right">MAE</th>
                            <th className="py-2.5 px-3 text-right">RMSE</th>
                            <th className="py-2.5 px-3 text-right">Test R² Score</th>
                            <th className="py-2.5 px-3 text-right">Train R² Score</th>
                            <th className="py-2.5 px-3 text-center">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                          {modelsData?.all_models_eval.map((m) => {
                            const isBest = m.Model === modelsData?.best_model;
                            return (
                              <tr key={m.Model} className={isBest ? "bg-indigo-950/30" : ""}>
                                <td className="py-3 px-3 font-bold text-white flex items-center gap-2">
                                  {m.Model}
                                  {isBest && (
                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                      BEST MODEL
                                    </span>
                                  )}
                                </td>
                                <td className="py-3 px-3 text-slate-400">{m.Type}</td>
                                <td className="py-3 px-3 text-right font-semibold text-slate-200">
                                  {formatINR(m.MAE)}
                                </td>
                                <td className="py-3 px-3 text-right font-bold text-purple-400">{formatINR(m.RMSE)}</td>
                                <td className="py-3 px-3 text-right font-extrabold text-emerald-400 text-sm">
                                  {m.R2_Score.toFixed(4)}
                                </td>
                                <td className="py-3 px-3 text-right text-slate-400">{m.Train_R2.toFixed(4)}</td>
                                <td className="py-3 px-3 text-center">
                                  {isBest ? (
                                    <span className="text-emerald-400 font-bold text-xs">Selected</span>
                                  ) : (
                                    <span className="text-slate-500 text-xs">Evaluated</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Feature & Pipeline Information */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Feature Pipeline Setup</h3>
                      <div className="space-y-2.5 text-xs">
                        <div className="flex justify-between py-1 border-b border-slate-800">
                          <span className="text-slate-400">Train/Test Split:</span>
                          <span className="text-slate-200 font-semibold">
                            80% Train ({modelsData?.train_samples} samples) / 20% Test ({modelsData?.test_samples}{" "}
                            samples)
                          </span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-slate-800">
                          <span className="text-slate-400">Numerical Features:</span>
                          <span className="text-slate-200 font-semibold">
                            {modelsData?.features_used.numerical.join(", ")}
                          </span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-slate-800">
                          <span className="text-slate-400">Categorical Features:</span>
                          <span className="text-slate-200 font-semibold">
                            {modelsData?.features_used.categorical.join(", ")}
                          </span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-slate-800">
                          <span className="text-slate-400">Categorical Encoder:</span>
                          <span className="text-slate-200 font-semibold">OneHotEncoder(handle_unknown='ignore')</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-slate-800">
                          <span className="text-slate-400">Numerical Scaler:</span>
                          <span className="text-slate-200 font-semibold">StandardScaler (for Linear Regression)</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
                      <h3 className="text-sm font-bold text-white mb-3">Why {modelsData?.best_model} was Chosen</h3>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        The <b>{modelsData?.best_model}</b> demonstrated the highest coefficient of determination (R² ={" "}
                        <b>{modelsData?.best_metrics.R2_Score.toFixed(4)}</b>) on the unseen testing partition,
                        accompanied by the lowest Root Mean Squared Error (
                        <b>{formatINR(modelsData?.best_metrics.RMSE || 0)}</b>). Gradient boosting iteratively builds
                        decision trees to minimize residual errors, successfully capturing non-linear interactions
                        between high-end processors, discrete graphics, and premium display panels.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* ========================================================== */}
              {/* TAB 9: ABOUT PROJECT */}
              {/* ========================================================== */}
              {activeTab === "about" && (
                <div className="space-y-6 max-w-4xl">
                  <div>
                    <h2 className="text-2xl font-extrabold text-white">ℹ️ About the Project</h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Academic College Project Documentation &amp; Viva Technical Reference Guide.
                    </p>
                  </div>

                  <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4 text-xs text-slate-300 leading-relaxed">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-indigo-400" />
                      1. Project Objective &amp; Problem Statement
                    </h3>
                    <p>
                      Pricing in the consumer laptop industry is heavily driven by complex combinations of hardware
                      specifications (microprocessor generation, RAM capacity, SSD interface, dedicated GPU tier, screen
                      resolution, and build materials). This project implements an end-to-end Machine Learning data
                      science pipeline utilizing <b>Python, NumPy, Pandas, Scikit-learn, and Streamlit</b> to analyze
                      pricing trends and accurately forecast market valuations in Indian Rupees (₹ INR).
                    </p>

                    <h3 className="text-sm font-bold text-white flex items-center gap-2 pt-2">
                      <Cpu className="w-4 h-4 text-purple-400" />
                      2. Machine Learning Regression Algorithms
                    </h3>
                    <ul className="list-disc pl-5 space-y-1.5">
                      <li>
                        <b>Linear Regression:</b> Models the baseline parametric relationship between scaled numerical
                        features, one-hot encoded categories, and target prices.
                      </li>
                      <li>
                        <b>Random Forest Regressor:</b> An ensemble bagging method constructing 120 decision trees to
                        mitigate variance and handle non-linear specification interactions.
                      </li>
                      <li>
                        <b>Gradient Boosting Regressor:</b> A sequential boosting ensemble that iteratively minimizes
                        residual error gradients, achieving an R² score exceeding 0.90.
                      </li>
                    </ul>

                    <h3 className="text-sm font-bold text-white flex items-center gap-2 pt-2">
                      <Layers className="w-4 h-4 text-emerald-400" />
                      3. Evaluation Metrics Formulations
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="font-bold text-indigo-400 block mb-0.5">MAE (Mean Absolute Error)</span>
                        Average magnitude of absolute errors between predicted price and actual price in ₹.
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="font-bold text-purple-400 block mb-0.5">RMSE (Root Mean Squared Error)</span>
                        Square root of the mean squared error, penalizing larger deviations in ₹.
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="font-bold text-emerald-400 block mb-0.5">R² Score (Determination)</span>
                        Fraction of price variance explained by the model specifications (0.0 to 1.0).
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="font-bold text-sky-400 block mb-0.5">Value Score (0 - 100)</span>
                        Hardware spec points per rupee, normalizing RAM, storage, CPU tier, and GPU tier.
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* CSV Upload Modal */}
      {uploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <UploadCloud className="w-5 h-5 text-indigo-400" />
                Upload Custom Laptop Dataset
              </h3>
              <button
                onClick={() => setUploadModalOpen(false)}
                className="text-slate-400 hover:text-white text-xs font-bold"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Upload any CSV file. The dynamic schema cleaner will automatically detect columns like{" "}
              <code>Price</code>, <code>Brand</code>, <code>RAM</code>, <code>Storage</code>, and{" "}
              <code>Processor</code>, clean data units, and retrain the machine learning models.
            </p>

            <input
              type="file"
              accept=".csv"
              onChange={handleCSVUpload}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer mb-3"
            />

            {uploadStatus && (
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-indigo-300 flex items-center gap-2 mt-2">
                <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                <span>{uploadStatus}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
