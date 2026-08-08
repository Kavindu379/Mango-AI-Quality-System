import React, { useState, useEffect, useRef } from 'react';
import {
  Scan,
  Cpu,
  Sparkles,
  DollarSign,
  Clock,
  AlertCircle,
  CheckCircle2,
  Layers,
  ShieldCheck,
  BarChart3,
  Upload,
  Camera,
  TrendingDown,
  Sun,
  Moon,
  Smartphone,
  X,
  Aperture
} from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:5000/api`;

function App() {
  const [theme, setTheme] = useState(localStorage.getItem('app-theme') || 'dark');
  const [activeTab, setActiveTab] = useState('scanner');
  const [basePrice, setBasePrice] = useState(300);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleName, setSampleName] = useState('sample_ripe_mango.jpg');
  const [sampleList, setSampleList] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(`${API_BASE}/samples/sample_ripe_mango.jpg`);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);

  // Webcam Camera Modal State for Desktop Webcams
  const [isCameraModalOpen, setIsCameraModalOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Apply Theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Fetch Samples & Check Flask Backend Health
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setBackendOnline(data.status === 'online'))
      .catch(() => setBackendOnline(false));

    fetch(`${API_BASE}/samples`)
      .then(res => res.json())
      .then(data => {
        if (data.samples && data.samples.length > 0) {
          setSampleList(data.samples);
          setSampleName(data.samples[0].filename);
          setPreviewUrl(`${API_BASE}/samples/${data.samples[0].filename}`);
          handleEvaluate(null, data.samples[0].filename, 300);
        } else {
          handleEvaluate(null, 'sample_ripe_mango.jpg', 300);
        }
      })
      .catch(err => {
        console.error("Error fetching sample list:", err);
        handleEvaluate(null, 'sample_ripe_mango.jpg', 300);
      });
  }, []);

  // Desktop Webcam Stream Launcher
  const openCameraModal = async () => {
    setIsCameraModalOpen(true);
    setCameraError(null);
    let stream = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { exact: 'environment' } }
      });
    } catch (e1) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (e2) {
        console.error("Camera access failed:", e2);
        setCameraError("Browser blocked live video stream over HTTP. Use 'Snap Live Photo' button to open your phone's camera directly!");
        return;
      }
    }

    if (stream) {
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    }
  };

  const closeCameraModal = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setIsCameraModalOpen(false);
    setCameraError(null);
  };

  const captureWebcamPhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        if (blob) {
          const capturedFile = new File([blob], "camera_snapshot.jpg", { type: "image/jpeg" });
          setSelectedFile(capturedFile);
          setSampleName(null);
          setPreviewUrl(URL.createObjectURL(blob));
          handleEvaluate(capturedFile, null, basePrice);
          closeCameraModal();
        }
      }, 'image/jpeg');
    }
  };

  const handleEvaluate = (file, sample, price) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('base_price', price || basePrice);

    if (file) {
      formData.append('image', file);
    } else if (sample) {
      formData.append('sample_name', sample);
    }

    fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.success) {
          setResult(data);
        }
      })
      .catch(err => {
        setLoading(false);
        console.error('API Error:', err);
      });
  };

  // Direct Mobile Camera Capture Handler (Uses native smartphone camera app)
  const handleMobileCameraCapture = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setSampleName(null);
      setPreviewUrl(URL.createObjectURL(file));
      handleEvaluate(file, null, basePrice);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setSampleName(null);
      setPreviewUrl(URL.createObjectURL(file));
      handleEvaluate(file, null, basePrice);
    }
  };

  const handleSelectSample = (sampleFile) => {
    setSelectedFile(null);
    setSampleName(sampleFile);
    setPreviewUrl(`${API_BASE}/samples/${sampleFile}`);
    handleEvaluate(null, sampleFile, basePrice);
  };

  const handlePriceChange = (newPrice) => {
    setBasePrice(newPrice);
    handleEvaluate(selectedFile, sampleName, newPrice);
  };

  const getSpectrumMarkerPosition = () => {
    if (!result) return '50%';
    const cls = result.prediction.class_code;
    if (cls === 'Grade_B_Unripe') return '15%';
    if (cls === 'Grade_A_Ripe') return '50%';
    if (cls === 'Grade_C_Overripe') return '85%';
    return '50%';
  };

  const getSampleDisplayInfo = (filename, index) => {
    const fn = filename.toLowerCase();
    if (fn.includes('unripe') || fn.includes('green') || index === 1) {
      return { icon: '🍏', title: 'Grade B (Unripe)' };
    }
    if (fn.includes('overripe') || fn.includes('damaged') || fn.includes('spoiled') || index === 2) {
      return { icon: '🍂', title: 'Grade C (Overripe)' };
    }
    return { icon: '🥭', title: 'Grade A (Ripe)' };
  };

  return (
    <div className="app-container">
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* WEBCAM CAMERA MODAL OVERLAY (FOR DESKTOP) */}
      {isCameraModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(10px)',
          zIndex: 2000,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1rem'
        }}>
          <div style={{
            position: 'relative',
            width: '100%',
            maxWidth: '680px',
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--amber-primary)',
            borderRadius: '1.25rem',
            padding: '1.25rem',
            boxShadow: '0 25px 50px -12px rgba(245, 158, 11, 0.25)',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--amber-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Aperture size={20} /> Live Camera Scanner
              </h3>
              <button onClick={closeCameraModal} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                <X size={24} />
              </button>
            </div>

            {cameraError ? (
              <div style={{ padding: '2rem 1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '0.85rem', marginBottom: '1.25rem' }}>
                <AlertCircle size={32} color="var(--rose-primary)" style={{ margin: '0 auto 0.75rem' }} />
                <p style={{ fontSize: '0.9rem', color: 'var(--text-main)', marginBottom: '0.5rem' }}>{cameraError}</p>
              </div>
            ) : (
              <div style={{ position: 'relative', width: '100%', maxHeight: '400px', backgroundColor: '#000', borderRadius: '0.85rem', overflow: 'hidden', marginBottom: '1.25rem', border: '1px solid var(--border-subtle)' }}>
                <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: '100%', maxHeight: '400px', objectFit: 'cover' }} />
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '200px', height: '200px', border: '2px dashed rgba(245, 158, 11, 0.6)', borderRadius: '50%', pointerEvents: 'none' }} />
              </div>
            )}

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button onClick={closeCameraModal} style={{ background: 'rgba(255, 255, 255, 0.1)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', padding: '0.75rem 1.5rem', borderRadius: '0.75rem', fontWeight: 600, cursor: 'pointer' }}>
                Close
              </button>
              {!cameraError && (
                <button onClick={captureWebcamPhoto} style={{ background: 'linear-gradient(135deg, var(--amber-primary), #d97706)', color: '#fff', border: 'none', padding: '0.75rem 2rem', borderRadius: '0.75rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', boxShadow: '0 4px 15px var(--amber-glow)' }}>
                  <Camera size={18} /> Capture & Scan Mango
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Top Navbar */}
      <nav className="navbar">
        <div className="brand-logo">
          <span className="logo-icon">🥭</span>
          <div>
            <h1 className="brand-title">Mango AI Vision & Grading</h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Intelligent Decision Support Platform</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className="module-tag">CS22032</span>
          <div className="system-status-indicator">
            <span className="status-dot" />
            <span>{backendOnline ? 'Model Online' : 'Connecting...'}</span>
          </div>

          <button className="theme-toggle-btn" onClick={toggleTheme}>
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </nav>

      {/* Navigation Tabs */}
      <div className="nav-tabs">
        <button className={`tab-btn ${activeTab === 'scanner' ? 'active' : ''}`} onClick={() => setActiveTab('scanner')}>
          <Scan size={18} />
          <span>AI Scanner</span>
        </button>
        <button className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
          <BarChart3 size={18} />
          <span>CNN Metrics</span>
        </button>
        <button className={`tab-btn ${activeTab === 'rules' ? 'active' : ''}`} onClick={() => setActiveTab('rules')}>
          <Layers size={18} />
          <span>Rules Matrix</span>
        </button>
      </div>

      {/* TAB 1: SCANNER & AI INFERENCE */}
      {activeTab === 'scanner' && (
        <div className="main-grid">
          {/* Left Column: Image Input & Control */}
          <div className="pro-card">
            <div className="card-header">
              <div className="card-title-group">
                <div className="card-icon-wrap">
                  <Camera size={18} />
                </div>
                <h2 className="card-title-text">Mango Input & Price</h2>
              </div>
            </div>

            <div className="price-control-box">
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>Market Base Price (Grade A)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Benchmark price / kg</div>
              </div>
              <div className="price-input-group">
                <span style={{ color: 'var(--amber-primary)', fontWeight: 700 }}>Rs.</span>
                <input
                  type="number"
                  className="price-input"
                  value={basePrice}
                  onChange={(e) => handlePriceChange(Number(e.target.value))}
                  min="50"
                  max="5000"
                  step="25"
                />
              </div>
            </div>

            {/* Native Mobile Camera & Gallery Button Inputs */}
            <div className="action-buttons-grid">
              <label className="action-btn-camera" htmlFor="mobile-camera-input">
                <Camera size={18} />
                <span>Snap Live Photo</span>
                <input
                  id="mobile-camera-input"
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleMobileCameraCapture}
                  style={{ display: 'none' }}
                />
              </label>

              <label className="action-btn-upload" htmlFor="mango-file-input">
                <Upload size={18} />
                <span>Browse Gallery</span>
                <input
                  id="mango-file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </div>

            {/* Dynamic Preset Samples Selector */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
                Preset Demo Samples:
              </label>
              <div className="preset-pills-grid">
                {sampleList.length > 0 ? (
                  sampleList.map((item, idx) => {
                    const info = getSampleDisplayInfo(item.filename, idx);
                    return (
                      <button
                        key={idx}
                        className={`preset-pill ${sampleName === item.filename ? 'active' : ''}`}
                        onClick={() => handleSelectSample(item.filename)}
                      >
                        <span style={{ fontSize: '1.2rem' }}>{info.icon}</span>
                        <span>{info.title}</span>
                      </button>
                    );
                  })
                ) : (
                  <>
                    <button className={`preset-pill ${sampleName === 'sample_ripe_mango.jpg' ? 'active' : ''}`} onClick={() => handleSelectSample('sample_ripe_mango.jpg')}>
                      <span>🥭</span><span>Grade A (Ripe)</span>
                    </button>
                    <button className={`preset-pill ${sampleName === 'sample_unripe_mango.jpg' ? 'active' : ''}`} onClick={() => handleSelectSample('sample_unripe_mango.jpg')}>
                      <span>🍏</span><span>Grade B (Unripe)</span>
                    </button>
                    <button className={`preset-pill ${sampleName === 'sample_overripe_mango.jpg' ? 'active' : ''}`} onClick={() => handleSelectSample('sample_overripe_mango.jpg')}>
                      <span>🍂</span><span>Grade C (Overripe)</span>
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Image Preview Frame */}
            {previewUrl && (
              <div className="image-preview-frame">
                <img src={previewUrl} alt="Target Mango" />
              </div>
            )}
          </div>

          {/* Right Column: AI Output & Decision Engine */}
          <div className="pro-card">
            <div className="card-header">
              <div className="card-title-group">
                <div className="card-icon-wrap">
                  <Cpu size={18} />
                </div>
                <h2 className="card-title-text">Neural Network & Decision Support</h2>
              </div>
            </div>

            {loading ? (
              <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--amber-primary)' }}>
                <Sparkles size={36} className="spin-loader" style={{ margin: '0 auto 1rem' }} />
                <p style={{ fontWeight: 600, fontSize: '1rem' }}>Processing Neural Network Inference...</p>
              </div>
            ) : result ? (
              <div>
                <div className={`result-banner ${result.prediction.class_code === 'Grade_A_Ripe' ? 'result-banner-grade-a' : result.prediction.class_code === 'Grade_B_Unripe' ? 'result-banner-grade-b' : 'result-banner-grade-c'}`}>
                  <div>
                    <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', opacity: 0.8 }}>Quality Grade</div>
                    <div className="result-banner-text">{result.prediction.display_name}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.7rem', opacity: 0.8 }}>Confidence</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800 }}>{result.prediction.confidence_percentage}%</div>
                  </div>
                </div>

                <div className="spectrum-bar-wrap">
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                    <span>Unripe Green</span>
                    <span>Optimal Ripe</span>
                    <span>Overripe Damaged</span>
                  </div>
                  <div className="spectrum-gradient">
                    <div className="spectrum-marker" style={{ left: getSpectrumMarkerPosition() }} />
                  </div>
                </div>

                <div style={{ marginBottom: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <Sparkles size={15} color="var(--amber-primary)" />
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-secondary)' }}>
                      Computer Vision Features (OpenCV HSV)
                    </span>
                  </div>
                  <div className="features-grid">
                    <div className="feature-box">
                      <div className="feature-label">Ripe Yellow %</div>
                      <div className="feature-value" style={{ color: 'var(--amber-primary)' }}>{result.computer_vision_features.yellow_percentage}%</div>
                    </div>
                    <div className="feature-box">
                      <div className="feature-label">Unripe Green %</div>
                      <div className="feature-value" style={{ color: 'var(--emerald-primary)' }}>{result.computer_vision_features.green_percentage}%</div>
                    </div>
                    <div className="feature-box">
                      <div className="feature-label">Dark Spots %</div>
                      <div className="feature-value" style={{ color: 'var(--rose-primary)' }}>{result.computer_vision_features.dark_spots_percentage}%</div>
                    </div>
                  </div>
                </div>

                <div style={{ background: 'rgba(0, 0, 0, 0.05)', border: '1px solid var(--border-subtle)', borderRadius: '0.85rem', padding: '1rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--amber-primary)', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <ShieldCheck size={16} />
                    Rule-Based Expert System Recommendations
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
                    <div style={{ background: 'var(--bg-surface)', padding: '0.85rem', borderRadius: '0.65rem', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <DollarSign size={13} color="var(--amber-primary)" />
                        Selling Price
                      </div>
                      <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--amber-primary)', marginTop: '0.2rem' }}>
                        Rs. {result.rule_engine.recommended_price}
                        {result.rule_engine.discount_percentage > 0 && (
                          <span style={{ fontSize: '0.7rem', color: 'var(--rose-primary)', marginLeft: '0.35rem', fontWeight: 600 }}>
                            <TrendingDown size={11} inline /> {result.rule_engine.discount_percentage}% OFF
                          </span>
                        )}
                      </div>
                    </div>

                    <div style={{ background: 'var(--bg-surface)', padding: '0.85rem', borderRadius: '0.65rem', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <Clock size={13} color="#38bdf8" />
                        Remaining Shelf Life
                      </div>
                      <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#38bdf8', marginTop: '0.25rem' }}>
                        {result.rule_engine.estimated_shelf_life}
                      </div>
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.85rem' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--emerald-primary)', marginBottom: '0.2rem' }}>
                      Vendor Operational Strategy ({result.rule_engine.status_category}):
                    </div>
                    <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      {result.rule_engine.vendor_recommendation}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-muted)' }}>
                <AlertCircle size={32} style={{ margin: '0 auto 0.75rem' }} />
                <p>Upload a fruit image or take a photo to view prediction results.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: MODEL ARCHITECTURE & PERFORMANCE */}
      {activeTab === 'analytics' && (
        <div className="pro-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon-wrap">
                <BarChart3 size={18} />
              </div>
              <h2 className="card-title-text">Neural Network Metrics & Layers</h2>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Training Accuracy</div>
              <div className="feature-value" style={{ color: 'var(--emerald-primary)', fontSize: '1.75rem' }}>95.00%</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Epoch 15 Performance</div>
            </div>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Final Loss Value</div>
              <div className="feature-value" style={{ color: 'var(--amber-primary)', fontSize: '1.75rem' }}>0.1600</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>CrossEntropyLoss</div>
            </div>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Model Weights</div>
              <div className="feature-value" style={{ color: '#38bdf8', fontSize: '1.2rem' }}>mango_model.pth</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>PyTorch State Dict</div>
            </div>
          </div>

          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem', color: 'var(--amber-primary)' }}>
            CNN Layer Specs (`MangoCNN`)
          </h3>
          <table className="pro-table">
            <thead>
              <tr>
                <th>Layer Block</th>
                <th>Operation</th>
                <th>Output Specs</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Input Layer</td>
                <td>Image Standardizer</td>
                <td>(B, 3, 224, 224)</td>
                <td>Standardizes RGB resolution & normalizes pixels</td>
              </tr>
              <tr>
                <td>Conv Block 1</td>
                <td>Conv2D (32 filters) + BatchNorm + MaxPool</td>
                <td>(B, 32, 112, 112)</td>
                <td>Extracts edge and color boundaries</td>
              </tr>
              <tr>
                <td>Conv Block 2</td>
                <td>Conv2D (64 filters) + BatchNorm + MaxPool</td>
                <td>(B, 64, 56, 56)</td>
                <td>Extracts surface texture & decay spot patterns</td>
              </tr>
              <tr>
                <td>Conv Block 3</td>
                <td>Conv2D (128 filters) + BatchNorm + MaxPool</td>
                <td>(B, 128, 28, 28)</td>
                <td>Extracts complex ripeness visual patterns</td>
              </tr>
              <tr>
                <td>Classifier Head</td>
                <td>AdaptiveAvgPool + Linear(128) + Dropout(0.3) + Linear(3)</td>
                <td>(B, 3)</td>
                <td>Softmax probabilities across Grade A, B, and C</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 3: RULE-BASED EXPERT SYSTEM MATRIX */}
      {activeTab === 'rules' && (
        <div className="pro-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon-wrap">
                <Layers size={18} />
              </div>
              <h2 className="card-title-text">Rule-Based Expert System Matrix</h2>
            </div>
          </div>

          <p style={{ color: 'var(--text-secondary)', marginBottom: '1.25rem', fontSize: '0.875rem', lineHeight: 1.5 }}>
            The expert inference engine evaluates predicted Neural Network classes and confidence percentages against standard supply-chain rules to ensure explainable decision-making.
          </p>

          <table className="pro-table">
            <thead>
              <tr>
                <th>Grade</th>
                <th>Ripeness</th>
                <th>Price Rule</th>
                <th>Shelf Life</th>
                <th>Operational Strategy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--emerald-primary)' }}>Grade A</td>
                <td>Ripe / Fresh 🥭</td>
                <td>100% Market Price</td>
                <td>3 to 5 Days</td>
                <td>Front counter display. Ideal for immediate sale. Store at 15°C–18°C.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--amber-primary)' }}>Grade B</td>
                <td>Unripe / Green 🍏</td>
                <td>90% Base Price</td>
                <td>7 to 10 Days</td>
                <td>Store at room temp (22°C–25°C) to ripen. Re-grade in 3 days.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 700, color: 'var(--rose-primary)' }}>Grade C</td>
                <td>Overripe / Damaged 🍂</td>
                <td>50% Discount</td>
                <td>1 Day</td>
                <td>Immediate clearance discount or transfer to juice processing. Isolate stock.</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <footer className="ai-concepts-footer">
        <div className="concept-item">
          <h4>1. Deep Learning CNN</h4>
          <p>3-block PyTorch Neural Network (`mango_model.pth`) trained for spatial feature classification.</p>
        </div>
        <div className="concept-item">
          <h4>2. Computer Vision (OpenCV)</h4>
          <p>HSV color space conversion calculating yellow ripeness, green immaturity, and dark spot decay ratios.</p>
        </div>
        <div className="concept-item">
          <h4>3. Rule-Based Expert System</h4>
          <p>Knowledge-based inference engine generating explainable price discounts, shelf life days, and storage guidance.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
