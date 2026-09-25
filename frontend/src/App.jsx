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
  X,
  AlertTriangle,
  Target
} from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:5000/api`;

// Currency Symbols Mapping
const CURRENCIES = {
  LKR: { symbol: 'Rs.', label: 'Rs. (LKR)' },
  USD: { symbol: '$', label: '$ (USD)' },
  INR: { symbol: '₹', label: '₹ (INR)' },
  EUR: { symbol: '€', label: '€ (EUR)' }
};

// Helper: RGB to HSV conversion
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, v = max;
  const d = max - min;
  s = max === 0 ? 0 : d / max;

  if (max === min) {
    h = 0;
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h * 360, s * 100, v * 100];
}

// Client-Side Image Compression (Max 1024x1024 to speed up network transfer)
function compressImageBeforeUpload(file, maxWidth = 1024, maxHeight = 1024, quality = 0.88) {
  return new Promise((resolve) => {
    if (!file || !file.type.startsWith('image/')) {
      resolve(file);
      return;
    }

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = () => {
        let width = img.width;
        let height = img.height;

        if (width <= maxWidth && height <= maxHeight) {
          resolve(file);
          return;
        }

        if (width > height) {
          if (width > maxWidth) {
            height = Math.round((height * maxWidth) / width);
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width = Math.round((width * maxHeight) / height);
            height = maxHeight;
          }
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob((blob) => {
          if (blob) {
            const compressedFile = new File([blob], file.name || "compressed_mango.jpg", {
              type: 'image/jpeg',
              lastModified: Date.now(),
            });
            resolve(compressedFile);
          } else {
            resolve(file);
          }
        }, 'image/jpeg', quality);
      };
      img.onerror = () => resolve(file);
    };
    reader.onerror = () => resolve(file);
  });
}

function App() {
  const [theme, setTheme] = useState(localStorage.getItem('app-theme') || 'dark');
  const [activeTab, setActiveTab] = useState('scanner');
  const [currency, setCurrency] = useState('LKR');
  const [basePrice, setBasePrice] = useState(300);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleName, setSampleName] = useState('sample_ripe_mango.jpg');
  const [sampleList, setSampleList] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(`${API_BASE}/samples/sample_ripe_mango.jpg`);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);
  const [isDragging, setIsDragging] = useState(false);

  // Bounding Box Overlay Ref & Aspect Dimensions
  const previewImgRef = useRef(null);
  const [imageNaturalSize, setImageNaturalSize] = useState({ width: 0, height: 0 });
  const [imageDisplaySize, setImageDisplaySize] = useState({ width: 0, height: 0 });

  // Webcam Camera Modal State
  const [isCameraModalOpen, setIsCameraModalOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const liveCanvasRef = useRef(null);
  const animFrameId = useRef(null);
  const smoothBoxRef = useRef({ x: 0, y: 0, w: 0, h: 0, active: false });

  // Theme Sync
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Fetch Samples & Backend Health
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
      .catch(() => handleEvaluate(null, 'sample_ripe_mango.jpg', 300));
  }, []);

  // Update Image Dimensions for SVG Overlay
  const handleImageLoaded = (e) => {
    setImageNaturalSize({ width: e.target.naturalWidth, height: e.target.naturalHeight });
    setImageDisplaySize({ width: e.target.clientWidth, height: e.target.clientHeight });
  };

  useEffect(() => {
    const handleResize = () => {
      if (previewImgRef.current) {
        setImageDisplaySize({
          width: previewImgRef.current.clientWidth,
          height: previewImgRef.current.clientHeight
        });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Camera Binding & Tracking
  useEffect(() => {
    if (isCameraModalOpen && cameraStream && videoRef.current) {
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play().catch(e => console.warn("Video play error:", e));
    }
  }, [isCameraModalOpen, cameraStream]);

  const refreshCameraDevices = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const rawVideoInputs = devices.filter(d => d.kind === 'videoinput');
      const validInputs = rawVideoInputs.filter(d => !(d.label || '').toLowerCase().includes('droidcam'));
      const inputsToUse = validInputs.length > 0 ? validInputs : rawVideoInputs;
      setVideoDevices(inputsToUse);
      
      if (inputsToUse.length > 0 && !selectedDeviceId) {
        const preferredCam = inputsToUse.find(d => 
          d.label.toLowerCase().includes('iriun') || 
          d.label.toLowerCase().includes('usb') ||
          d.label.toLowerCase().includes('integrated') ||
          d.label.toLowerCase().includes('webcam')
        );
        setSelectedDeviceId(preferredCam ? preferredCam.deviceId : inputsToUse[0].deviceId);
      }
    } catch (e) {
      console.error("Camera enumeration error:", e);
    }
  };

  // Ultra-Fast Live Fruit Tracking
  useEffect(() => {
    if (isCameraModalOpen && cameraStream) {
      const trackObjectInVideo = () => {
        if (videoRef.current && liveCanvasRef.current && videoRef.current.readyState === 4) {
          const video = videoRef.current;
          const canvas = liveCanvasRef.current;
          const ctx = canvas.getContext('2d');
          
          canvas.width = video.videoWidth || 1280;
          canvas.height = video.videoHeight || 720;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = frame.data;
          
          const matchedPoints = [];
          const step = 8;
          
          for (let y = 0; y < canvas.height; y += step) {
            for (let x = 0; x < canvas.width; x += step) {
              const i = (y * canvas.width + x) * 4;
              const [h, s, v] = rgbToHsv(data[i], data[i + 1], data[i + 2]);
              // Sync with backend OpenCV masks: OpenCV Hue (0-180) -> JS Hue (0-360)
              // Extremely tight thresholds for max accuracy (only vivid fruit colors, ignore screen glare)
              const isRipeYellow = (h >= 30 && h <= 60 && s >= 50 && v >= 40);
              const isUnripeGreen = (h >= 75 && h <= 140 && s >= 50 && v >= 40);
              
              if (isRipeYellow || isUnripeGreen) {
                matchedPoints.push({ x, y });
              }
            }
          }
          
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          if (matchedPoints.length > 25) {
            // Find Median instead of Mean to be completely immune to background noise outliers
            matchedPoints.sort((a, b) => a.x - b.x);
            const medianX = matchedPoints[Math.floor(matchedPoints.length / 2)].x;
            matchedPoints.sort((a, b) => a.y - b.y);
            const medianY = matchedPoints[Math.floor(matchedPoints.length / 2)].y;
            
            const validPoints = matchedPoints.filter(p => {
              const dx = p.x - medianX;
              const dy = p.y - medianY;
              // Tighter spatial radius based on median center
              return (dx * dx + dy * dy) < (canvas.width * canvas.width * 0.12);
            });
            
            if (validPoints.length > 18) {
              // Use 5th and 95th percentiles to aggressively tightly bound the core fruit mass
              validPoints.sort((a, b) => a.x - b.x);
              const minX = validPoints[Math.floor(validPoints.length * 0.05)].x;
              const maxX = validPoints[Math.floor(validPoints.length * 0.95)].x;
              
              validPoints.sort((a, b) => a.y - b.y);
              const minY = validPoints[Math.floor(validPoints.length * 0.05)].y;
              const maxY = validPoints[Math.floor(validPoints.length * 0.95)].y;
              
              const rawW = maxX - minX;
              const rawH = maxY - minY;
              
              if (rawW < canvas.width * 0.85 && rawH < canvas.height * 0.85 && rawW > 25 && rawH > 25) {
                const smooth = smoothBoxRef.current;
                if (!smooth.active) {
                  smooth.x = minX; smooth.y = minY; smooth.w = rawW; smooth.h = rawH; smooth.active = true;
                } else {
                  smooth.x += (minX - smooth.x) * 0.35;
                  smooth.y += (minY - smooth.y) * 0.35;
                  smooth.w += (rawW - smooth.w) * 0.35;
                  smooth.h += (rawH - smooth.h) * 0.35;
                }

                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 3.5;
                ctx.shadowColor = '#10b981';
                ctx.shadowBlur = 12;
                ctx.strokeRect(smooth.x, smooth.y, smooth.w, smooth.h);
                
                ctx.fillStyle = '#10b981';
                ctx.shadowBlur = 0;
                ctx.fillRect(smooth.x, smooth.y > 28 ? smooth.y - 26 : smooth.y, 160, 24);
                ctx.fillStyle = '#000000';
                ctx.font = 'bold 12px Inter, sans-serif';
                ctx.fillText('🥭 Target Mango Identified', smooth.x + 6, smooth.y > 28 ? smooth.y - 9 : smooth.y + 16);
              } else {
                smoothBoxRef.current.active = false;
              }
            } else {
              smoothBoxRef.current.active = false;
            }
          } else {
            smoothBoxRef.current.active = false;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const size = 110;
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
            ctx.lineWidth = 2;
            ctx.setLineDash([8, 8]);
            ctx.strokeRect(centerX - size/2, centerY - size/2, size, size);
            ctx.setLineDash([]);
            ctx.fillStyle = '#f59e0b';
            ctx.font = 'bold 13px Inter, sans-serif';
            ctx.fillText('🔍 Place Mango Inside Target', centerX - 80, centerY - size/2 - 10);
          }
        }
        animFrameId.current = requestAnimationFrame(trackObjectInVideo);
      };
      
      animFrameId.current = requestAnimationFrame(trackObjectInVideo);
    }
    return () => {
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
    };
  }, [isCameraModalOpen, cameraStream]);

  const startCameraWithDevice = async (deviceIdToUse) => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      setCameraStream(null);
    }
    smoothBoxRef.current = { x: 0, y: 0, w: 0, h: 0, active: false };
    setCameraError(null);
    let stream = null;

    try {
      if (deviceIdToUse) {
        stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceIdToUse } } });
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }
    } catch (e1) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (e2) {
        setCameraError("Could not access camera. Please check browser permissions.");
        return;
      }
    }

    if (stream) {
      setCameraStream(stream);
      refreshCameraDevices();
    }
  };

  const openCameraModal = async () => {
    setIsCameraModalOpen(true);
    await startCameraWithDevice(selectedDeviceId);
  };

  const handleDeviceChange = (e) => {
    const newDeviceId = e.target.value;
    setSelectedDeviceId(newDeviceId);
    startCameraWithDevice(newDeviceId);
  };

  const closeCameraModal = () => {
    if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
    smoothBoxRef.current = { x: 0, y: 0, w: 0, h: 0, active: false };
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      setCameraStream(null);
    }
    setIsCameraModalOpen(false);
    setCameraError(null);
  };

  const captureWebcamPhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const smooth = smoothBoxRef.current;
      
      const videoW = video.videoWidth || 1280;
      const videoH = video.videoHeight || 720;
      const liveCanvas = liveCanvasRef.current;
      const displayW = liveCanvas ? liveCanvas.width : videoW;
      const displayH = liveCanvas ? liveCanvas.height : videoH;
      const scaleX = videoW / displayW;
      const scaleY = videoH / displayH;

      const ctx = canvas.getContext('2d');

      if (smooth && smooth.active && smooth.w > 25 && smooth.h > 25) {
        const padX = smooth.w * 0.12;
        const padY = smooth.h * 0.12;
        const cropX = Math.max(0, (smooth.x - padX) * scaleX);
        const cropY = Math.max(0, (smooth.y - padY) * scaleY);
        const cropW = Math.min(videoW - cropX, (smooth.w + padX * 2) * scaleX);
        const cropH = Math.min(videoH - cropY, (smooth.h + padY * 2) * scaleY);

        canvas.width = cropW;
        canvas.height = cropH;
        ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
      } else {
        canvas.width = videoW;
        canvas.height = videoH;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      
      canvas.toBlob((blob) => {
        if (blob) {
          const capturedFile = new File([blob], "tracked_mango.jpg", { type: "image/jpeg" });
          setSelectedFile(capturedFile);
          setSampleName(null);
          setPreviewUrl(URL.createObjectURL(blob));
          handleEvaluate(capturedFile, null, basePrice);
          closeCameraModal();
        }
      }, 'image/jpeg', 0.95);
    }
  };

  const handleSnapLivePhotoClick = () => {
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (isMobile) {
      const mobileInput = document.getElementById('mobile-camera-input');
      if (mobileInput) mobileInput.click();
    } else {
      openCameraModal();
    }
  };

  const handleEvaluate = async (file, sample, price) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('base_price', price || basePrice);

    if (file) {
      const compressedFile = await compressImageBeforeUpload(file);
      formData.append('image', compressedFile);
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
        console.error('API Prediction Error:', err);
      });
  };

  const processUploadedFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setSampleName(null);
      setPreviewUrl(URL.createObjectURL(file));
      handleEvaluate(file, null, basePrice);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const handleMobileCameraCapture = (e) => {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0]);
    }
  };

  const handleFileUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0]);
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
    if (fn.includes('unripe') || fn.includes('green')) {
      return { icon: '🍏', title: 'Grade B (Unripe)' };
    }
    if (fn.includes('overripe') || fn.includes('damaged') || fn.includes('rot')) {
      return { icon: '🍂', title: 'Grade C (Overripe)' };
    }
    if (fn.includes('ripe') || fn.includes('grade_a')) {
      return { icon: '🥭', title: 'Grade A (Ripe)' };
    }
    // Fallback if the file is just named 1.jpeg, 2.jpeg, etc.
    return { icon: '🖼️', title: filename };
  };

  const currSymbol = CURRENCIES[currency]?.symbol || 'Rs.';

  return (
    <div className="app-container">
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* FULL HD CAMERA MODAL */}
      {isCameraModalOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.88)', backdropFilter: 'blur(12px)',
          zIndex: 2000, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1rem'
        }}>
          <div style={{
            position: 'relative', width: '100%', maxWidth: '740px',
            backgroundColor: 'var(--bg-surface)', border: '1px solid var(--amber-primary)',
            borderRadius: '1.25rem', padding: '1.25rem', boxShadow: '0 25px 50px -12px rgba(245, 158, 11, 0.3)', textAlign: 'center'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--amber-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Target size={20} className="spin-loader" /> Full HD Camera Scanner
              </h3>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {videoDevices.length > 0 && (
                  <select 
                    value={selectedDeviceId} 
                    onChange={handleDeviceChange}
                    style={{
                      background: 'var(--bg-card-inner)', color: 'var(--amber-primary)',
                      border: '1px solid var(--amber-primary)', borderRadius: '0.5rem',
                      padding: '0.35rem 0.65rem', fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer'
                    }}
                  >
                    {videoDevices.map((dev, idx) => (
                      <option key={dev.deviceId || idx} value={dev.deviceId}>
                        {dev.label || `Camera ${idx + 1}`}
                      </option>
                    ))}
                  </select>
                )}
                <button onClick={closeCameraModal} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                  <X size={24} />
                </button>
              </div>
            </div>

            {cameraError ? (
              <div style={{ padding: '2rem 1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '0.85rem', marginBottom: '1.25rem' }}>
                <AlertCircle size={32} color="var(--rose-primary)" style={{ margin: '0 auto 0.75rem' }} />
                <p style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>{cameraError}</p>
              </div>
            ) : (
              <div style={{ position: 'relative', width: '100%', maxHeight: '420px', backgroundColor: '#000', borderRadius: '0.85rem', overflow: 'hidden', marginBottom: '1.25rem' }}>
                <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', maxHeight: '420px', objectFit: 'cover' }} />
                <canvas ref={liveCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', objectFit: 'cover' }} />
              </div>
            )}

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button onClick={closeCameraModal} style={{ background: 'rgba(255, 255, 255, 0.1)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', padding: '0.75rem 1.5rem', borderRadius: '0.75rem', fontWeight: 600, cursor: 'pointer' }}>
                Close
              </button>
              {!cameraError && (
                <button onClick={captureWebcamPhoto} style={{ background: 'linear-gradient(135deg, var(--amber-primary), #d97706)', color: '#fff', border: 'none', padding: '0.75rem 2rem', borderRadius: '0.75rem', fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', boxShadow: '0 4px 15px var(--amber-glow)' }}>
                  <Camera size={18} /> Capture Focused Mango Box
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TOP NAVBAR */}
      <nav className="navbar">
        <div className="brand-logo">
          <div className="logo-icon-wrap">🥭</div>
          <div>
            <h1 className="brand-title">Mango AI Vision & Grading</h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Intelligent Decision Support Platform</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
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

      {/* NAVIGATION TABS */}
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
                <h2 className="card-title-text">Mango Input & Pricing Controller</h2>
              </div>
            </div>

            {/* Price & Currency Bar */}
            <div className="price-control-box">
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>Market Base Price (Grade A)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Benchmark price / kg</div>
              </div>
              <div className="price-input-group">
                <select 
                  className="currency-select"
                  value={currency} 
                  onChange={(e) => setCurrency(e.target.value)}
                >
                  {Object.keys(CURRENCIES).map(code => (
                    <option key={code} value={code}>{code}</option>
                  ))}
                </select>
                <input
                  type="number"
                  className="price-input"
                  value={basePrice}
                  onChange={(e) => handlePriceChange(Number(e.target.value))}
                  min="1"
                  max="10000"
                  step="10"
                />
              </div>
            </div>

            {/* Drag and Drop Zone */}
            <div 
              className={`dropzone-container ${isDragging ? 'is-dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload size={28} color="var(--amber-primary)" style={{ margin: '0 auto 0.4rem' }} />
              <div style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-main)' }}>
                Drag & Drop Fruit Image Here
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                Supports JPG, PNG, WebP smartphone photos
              </div>
            </div>

            <div className="action-buttons-grid">
              <button className="action-btn-camera" onClick={handleSnapLivePhotoClick}>
                <Camera size={18} />
                <span>Snap Live Photo</span>
              </button>

              <input
                id="mobile-camera-input"
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleMobileCameraCapture}
                style={{ display: 'none' }}
              />

              <label className="action-btn-camera" htmlFor="mango-file-input" style={{ background: 'var(--bg-card-inner)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', boxShadow: 'none' }}>
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

            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
                Preset Test Samples:
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

            {/* PREVIEW FRAME */}
            {previewUrl && (
              <div className="image-preview-wrapper">
                <img 
                  ref={previewImgRef} 
                  src={previewUrl} 
                  alt="Target Mango" 
                  onLoad={handleImageLoaded}
                />
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
                <h2 className="card-title-text">Neural Network & Decision Engine</h2>
              </div>
            </div>

            {loading ? (
              <div style={{ textAlign: 'center', padding: '3.5rem 0', color: 'var(--amber-primary)' }}>
                <Sparkles size={40} className="spin-loader" style={{ margin: '0 auto 1rem' }} />
                <p style={{ fontWeight: 700, fontSize: '1.05rem' }}>Processing Neural Network Inference...</p>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>Auto-Cropping Bounding Box & Extracting HSV Ratios</p>
              </div>
            ) : result ? (
              <div>
                {(!result.is_valid_mango || result.prediction.class_code === 'Non_Mango') && (
                  <div style={{
                    padding: '1rem 1.25rem',
                    background: 'rgba(239, 68, 68, 0.15)',
                    border: '1px solid var(--rose-primary)',
                    borderRadius: '0.85rem',
                    marginBottom: '1.25rem',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.85rem'
                  }}>
                    <AlertTriangle size={26} color="var(--rose-primary)" style={{ flexShrink: 0, marginTop: '0.1rem' }} />
                    <div>
                      <h4 style={{ color: 'var(--rose-primary)', fontWeight: 800, fontSize: '0.95rem', marginBottom: '0.2rem' }}>
                        Invalid Object Detected 🚫
                      </h4>
                      <p style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: 1.4 }}>
                        {result.rejection_reason || "The scanned item does not match Mango visual characteristics or color features. Please scan a valid Mango (Ripe, Unripe, or Overripe)."}
                      </p>
                    </div>
                  </div>
                )}

                <div className={`result-banner ${
                  result.prediction.class_code === 'Grade_A_Ripe' ? 'result-banner-grade-a' : 
                  result.prediction.class_code === 'Grade_B_Unripe' ? 'result-banner-grade-b' : 
                  'result-banner-grade-c'
                }`}>
                  <div>
                    <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', opacity: 0.85 }}>Quality Classification</div>
                    <div className="result-banner-text">{result.prediction.display_name}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.7rem', opacity: 0.85 }}>Confidence</div>
                    <div style={{ fontSize: '1.45rem', fontWeight: 800 }}>{result.prediction.confidence_percentage}%</div>
                  </div>
                </div>

                {result.is_valid_mango && result.prediction.class_code !== 'Non_Mango' && (
                  <div className="spectrum-bar-wrap">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>
                      <span>Unripe Green</span>
                      <span>Optimal Ripe</span>
                      <span>Overripe Damaged</span>
                    </div>
                    <div className="spectrum-gradient">
                      <div className="spectrum-marker" style={{ left: getSpectrumMarkerPosition() }} />
                    </div>
                  </div>
                )}

                {/* Computer Vision OpenCV HSV Features */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <Sparkles size={15} color="var(--amber-primary)" />
                    <span style={{ fontSize: '0.8rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>
                      Computer Vision Color Analysis (HSV Ratios)
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

                {/* All Classes Probability Breakdown Card */}
                {result.prediction.class_probabilities && Object.keys(result.prediction.class_probabilities).length > 0 && (
                  <div className="class-prob-card">
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      All Class Confidence Probabilities
                    </div>
                    {Object.entries(result.prediction.class_probabilities).map(([code, prob]) => (
                      <div key={code} className="prob-row">
                        <div className="prob-meta">
                          <span>{code.replace('_', ' ')}</span>
                          <span>{prob}%</span>
                        </div>
                        <div className="prob-track">
                          <div 
                            className="prob-fill" 
                            style={{ 
                              width: `${prob}%`,
                              background: code === 'Grade_A_Ripe' ? 'var(--emerald-primary)' :
                                          code === 'Grade_B_Unripe' ? 'var(--amber-primary)' :
                                          code === 'Grade_C_Overripe' ? 'var(--rose-primary)' : 'var(--text-muted)'
                            }} 
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Rule Engine Guidance Card */}
                <div style={{ background: 'var(--bg-card-inner)', border: '1px solid var(--border-subtle)', borderRadius: '0.95rem', padding: '1.1rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--amber-primary)', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <ShieldCheck size={16} />
                    Rule-Based Expert System Recommendations
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
                    <div style={{ background: 'var(--bg-surface)', padding: '0.85rem', borderRadius: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <DollarSign size={13} color="var(--amber-primary)" />
                        Selling Price
                      </div>
                      <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--amber-primary)', marginTop: '0.2rem' }}>
                        {currSymbol} {result.rule_engine.recommended_price}
                        {result.rule_engine.discount_percentage > 0 && (
                          <span style={{ fontSize: '0.7rem', color: 'var(--rose-primary)', marginLeft: '0.35rem', fontWeight: 700 }}>
                            <TrendingDown size={11} inline /> {result.rule_engine.discount_percentage}% OFF
                          </span>
                        )}
                      </div>
                    </div>

                    <div style={{ background: 'var(--bg-surface)', padding: '0.85rem', borderRadius: '0.75rem', border: '1px solid var(--border-subtle)' }}>
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
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--emerald-primary)', marginBottom: '0.2rem' }}>
                      Vendor Operational Strategy ({result.rule_engine.status_category}):
                    </div>
                    <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                      {result.rule_engine.vendor_recommendation}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '3.5rem 0', color: 'var(--text-muted)' }}>
                <AlertCircle size={36} style={{ margin: '0 auto 0.75rem' }} />
                <p style={{ fontSize: '0.9rem' }}>Upload a fruit photo or select a sample to view decision results.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: MODEL METRICS */}
      {activeTab === 'analytics' && (
        <div className="pro-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon-wrap">
                <BarChart3 size={18} />
              </div>
              <h2 className="card-title-text">Neural Network Architecture & Metrics</h2>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Training Accuracy</div>
              <div className="feature-value" style={{ color: 'var(--emerald-primary)', fontSize: '1.75rem' }}>95.50%</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Epoch 15 Performance</div>
            </div>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Final Loss Value</div>
              <div className="feature-value" style={{ color: 'var(--amber-primary)', fontSize: '1.75rem' }}>0.1600</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>CrossEntropyLoss</div>
            </div>
            <div className="feature-box" style={{ padding: '1rem', textAlign: 'left' }}>
              <div className="feature-label">Model Architecture</div>
              <div className="feature-value" style={{ color: '#38bdf8', fontSize: '1.2rem' }}>MobileNetV2</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>PyTorch State Dict</div>
            </div>
          </div>

          <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '0.75rem', color: 'var(--amber-primary)' }}>
            Neural Network Layer Specifications
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
                <td>Base Backbone</td>
                <td>MobileNetV2 Features (ImageNet Weights)</td>
                <td>(B, 1280, 7, 7)</td>
                <td>Depthwise Separable Convolutions & Bottlenecks</td>
              </tr>
              <tr>
                <td>Classifier Head</td>
                <td>Dropout(0.3) + Linear(1280, 4)</td>
                <td>(B, 4)</td>
                <td>Softmax probabilities across 4 target classes</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 3: RULE-BASED EXPERT MATRIX */}
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
            The expert inference engine evaluates predicted Neural Network classes and confidence percentages against supply-chain rules to ensure explainable decision-making.
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
                <td style={{ fontWeight: 800, color: 'var(--emerald-primary)' }}>Grade A</td>
                <td>Ripe / Fresh 🥭</td>
                <td>100% Market Price</td>
                <td>3 to 5 Days</td>
                <td>Front counter display. Ideal for immediate sale. Store at 15°C–18°C.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 800, color: 'var(--amber-primary)' }}>Grade B</td>
                <td>Unripe / Green 🍏</td>
                <td>90% Base Price</td>
                <td>7 to 10 Days</td>
                <td>Store at room temp (22°C–25°C) to ripen. Re-grade in 3 days.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 800, color: 'var(--rose-primary)' }}>Grade C</td>
                <td>Overripe / Damaged 🍂</td>
                <td>50% Discount</td>
                <td>1 Day</td>
                <td>Immediate clearance discount or transfer to juicing. Isolate stock.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 800, color: 'var(--rose-primary)' }}>Non_Mango</td>
                <td>Invalid Object 🚫</td>
                <td>N/A</td>
                <td>N/A</td>
                <td>⚠️ Non-Mango object detected. Rejects non-fruit items cleanly.</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* FOOTER */}
      <footer className="ai-concepts-footer">
        <div className="concept-item">
          <h4>1. Deep Learning CNN</h4>
          <p>4-class PyTorch Neural Network (`mango_model.pth`) for spatial feature classification & OOD detection.</p>
        </div>
        <div className="concept-item">
          <h4>2. Computer Vision (OpenCV)</h4>
          <p>HSV color space conversion calculating yellow ripeness, green immaturity, and dark spot decay ratios.</p>
        </div>
        <div className="concept-item">
          <h4>3. Rule-Based Expert System</h4>
          <p>Knowledge-based inference engine generating explainable price discounts, shelf life, and storage guidance.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
