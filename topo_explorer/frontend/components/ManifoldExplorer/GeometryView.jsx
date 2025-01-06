import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const GeometryView = ({ manifoldData, trajectory, isPlaying }) => {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameRef = useRef(null);
  const [error, setError] = useState(null);
  const isFirefox = typeof InstallTrigger !== 'undefined';

  useEffect(() => {
    let isActive = true; 

    const init = () => {
      try {
        if (!canvasRef.current) return;

        sceneRef.current = new THREE.Scene();
        sceneRef.current.background = new THREE.Color(0xf0f0f0);

        const aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
        cameraRef.current = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        cameraRef.current.position.set(5, 5, 5);
        cameraRef.current.lookAt(0, 0, 0);

        rendererRef.current = new THREE.WebGLRenderer({
          canvas: canvasRef.current,
          antialias: !isFirefox,
          alpha: true,
          precision: isFirefox ? 'lowp' : 'mediump',
          powerPreference: isFirefox ? 'low-power' : 'default',
          failIfMajorPerformanceCaveat: false,
          forceWebGL1: isFirefox
        });

        rendererRef.current.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
        rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        if (cameraRef.current && rendererRef.current) {
          controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
          
          controlsRef.current.enableDamping = true;
          controlsRef.current.dampingFactor = 0.1; 
          controlsRef.current.rotateSpeed = 0.8; 
          
          controlsRef.current.enableZoom = true;
          controlsRef.current.zoomSpeed = 0.8;
          controlsRef.current.minDistance = 4;
          controlsRef.current.maxDistance = 20;
          
          controlsRef.current.enablePan = true;
          controlsRef.current.screenSpacePanning = true;
          controlsRef.current.panSpeed = 0.8;
          
          controlsRef.current.minPolarAngle = Math.PI * 0.1; 
          controlsRef.current.maxPolarAngle = Math.PI * 0.9; 
          
          controlsRef.current.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN
          };
          
          controlsRef.current.enableKeys = true;
          controlsRef.current.keys = {
            LEFT: 'ArrowLeft',
            UP: 'ArrowUp',
            RIGHT: 'ArrowRight',
            BOTTOM: 'ArrowDown'
          };
        }        

        const ambientLight = new THREE.AmbientLight(0x404040, 1);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        sceneRef.current.add(ambientLight);
        sceneRef.current.add(directionalLight);

        const animate = () => {
          if (!isActive) return;
          animationFrameRef.current = requestAnimationFrame(animate);
          
          if (controlsRef.current) {
            controlsRef.current.update();
          }
          
          if (rendererRef.current && sceneRef.current && cameraRef.current) {
            rendererRef.current.render(sceneRef.current, cameraRef.current);
          }
        };
        animate();

      } catch (error) {
        console.error('Failed to initialize 3D viewer:', error);
        setError('Failed to initialize 3D viewer');
      }
    };

    init();

    return () => {
      isActive = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (sceneRef.current) {
        sceneRef.current.traverse((object) => {
          if (object.geometry) object.geometry.dispose();
          if (object.material) {
            if (Array.isArray(object.material)) {
              object.material.forEach(material => material.dispose());
            } else {
              object.material.dispose();
            }
          }
        });
      }
    };
  }, [isFirefox]);

  useEffect(() => {
    if (!sceneRef.current || !manifoldData) {
      console.log('Missing scene or manifold data:', { scene: !!sceneRef.current, manifoldData });
      return;
    }
    console.log('Updating manifold:', manifoldData.type);

    sceneRef.current.children = sceneRef.current.children.filter(child => {
      if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
        child.geometry.dispose();
        child.material.dispose();
        return false;
      }
      return true;
    });

    let geometry;
    let material;

    try {
      switch (manifoldData.type) {
        case 'sphere':
          geometry = new THREE.SphereGeometry(2, 32, 32);
          material = new THREE.MeshPhongMaterial({
            color: 0x156289,
            transparent: true,
            opacity: 0.8,
            shininess: 70,
            side: THREE.DoubleSide
          });
          break;

        case 'torus':
          geometry = new THREE.TorusGeometry(3, 1, 30, 30);
          material = new THREE.MeshPhongMaterial({
            color: 0x156289,
            transparent: true,
            opacity: 0.85,
            shininess: 60,
            side: THREE.DoubleSide
          });
          break;

        case 'mobius':
        case 'klein':
        case 'projective':
          if (manifoldData.surface) {
            const [x, y, z] = manifoldData.surface;
            const vertices = [];
            for (let i = 0; i < x.length; i++) {
              for (let j = 0; j < x[i].length; j++) {
                vertices.push(x[i][j], y[i][j], z[i][j]);
              }
            }
            geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.computeVertexNormals();
            
            material = new THREE.MeshPhongMaterial({
              color: 0x156289,
              transparent: true,
              opacity: 0.85,
              shininess: 50,
              side: THREE.DoubleSide
            });
          }
          break;

        case 'hyperbolic':
          geometry = new THREE.CircleGeometry(2, 64);
          material = new THREE.MeshBasicMaterial({
            color: 0x156289,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide,
            wireframe: true
          });
          break;
      }

      if (geometry && material) {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = 'manifold';
        sceneRef.current.add(mesh);

        if (manifoldData.type !== 'hyperbolic') {
          const wireframe = new THREE.WireframeGeometry(geometry);
          const wireframeMaterial = new THREE.LineBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.15
          });
          const wireframeMesh = new THREE.LineSegments(wireframe, wireframeMaterial);
          wireframeMesh.name = 'manifold-wireframe';
          sceneRef.current.add(wireframeMesh);
        }

        if (manifoldData.type === 'hyperbolic') {
          cameraRef.current.position.set(0, 0, 5);
        } else {
          cameraRef.current.position.set(4, 4, 4);
        }
        cameraRef.current.lookAt(0, 0, 0);
        controlsRef.current.update();
      }
    } catch (error) {
      console.error('Error creating manifold:', error);
      setError('Failed to create manifold visualization');
    }
  }, [manifoldData]);

  useEffect(() => {
    if (!cameraRef.current || !manifoldData) return;
    
    const resetCamera = () => {
      switch (manifoldData.type) {
        case 'hyperbolic':
          cameraRef.current.position.set(0, 0, 8);
          break;
        case 'torus':
          cameraRef.current.position.set(6, 6, 6);
          break;
        case 'klein':
        case 'mobius':
          cameraRef.current.position.set(5, 5, 8);
          break;
        default:
          cameraRef.current.position.set(5, 5, 5);
      }
      cameraRef.current.lookAt(0, 0, 0);
      
      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
        
        if (manifoldData.type === 'hyperbolic') {
          controlsRef.current.minDistance = 6;
          controlsRef.current.maxDistance = 15;
        } else {
          controlsRef.current.minDistance = 4;
          controlsRef.current.maxDistance = 20;
        }
      }
    };
  
    resetCamera();
  }, [manifoldData?.type]);
  
  const animate = () => {
    if (!isActive) return;
    
    animationFrameRef.current = requestAnimationFrame(animate);
    
    if (controlsRef.current) {
      controlsRef.current.update(); 
    }
    
    if (rendererRef.current && sceneRef.current && cameraRef.current) {
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }
  };

  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isPlaying;
      controlsRef.current.autoRotateSpeed = 2.0;
    }
  }, [isPlaying]);

  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height, false);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      
      rendererRef.current.setSize(width, height, false);
      rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      
      if (sceneRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
  
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="w-full h-96 relative">
      {error ? (
        <div className="w-full h-full flex items-center justify-center bg-red-50 text-red-600">
          {error}
        </div>
      ) : (
        <canvas 
          ref={canvasRef}
          className="w-full h-full"
          style={{ display: 'block' }}
        />
      )}
    </div>
  );
};

export default GeometryView;