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
  const isActiveRef = useRef(true);
  const [error, setError] = useState(null);

  // Initial setup
  useEffect(() => {
    const init = async () => {
      if (!canvasRef.current) return;

      try {
        // Scene setup
        sceneRef.current = new THREE.Scene();
        sceneRef.current.background = new THREE.Color(0xf5f5f5);

        // Camera setup
        const aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
        cameraRef.current = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        cameraRef.current.position.set(5, 5, 5);
        cameraRef.current.lookAt(0, 0, 0);

        // Renderer setup
        rendererRef.current = new THREE.WebGLRenderer({
          canvas: canvasRef.current,
          antialias: true,
          alpha: true,
          preserveDrawingBuffer: true, // Add this
          logarithmicDepthBuffer: true // Add this
        });
        rendererRef.current.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
        rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Controls setup
        if (cameraRef.current && rendererRef.current) {
          controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
          
          // Control settings
          controlsRef.current.enableDamping = false;
          controlsRef.current.autoRotate = false;
          controlsRef.current.rotateSpeed = 1.0;
          controlsRef.current.zoomSpeed = 1.0;
          controlsRef.current.panSpeed = 1.0;
          controlsRef.current.enableZoom = true;
          controlsRef.current.enablePan = true;
          controlsRef.current.enableRotate = true;
          controlsRef.current.minDistance = 3;
          controlsRef.current.maxDistance = 20;
        }

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); 
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0); 
        directionalLight.position.set(5, 5, 5);
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5); 
        directionalLight2.position.set(-5, -5, -5);
        sceneRef.current.add(ambientLight);
        sceneRef.current.add(directionalLight);
        sceneRef.current.add(directionalLight2);

        // Animation function
        const animate = () => {
          if (!isActiveRef.current) return;
          
          animationFrameRef.current = requestAnimationFrame(animate);
          
          if (controlsRef.current) {
            controlsRef.current.update();
          }
          
          if (rendererRef.current && sceneRef.current && cameraRef.current) {
            rendererRef.current.render(sceneRef.current, cameraRef.current);
          }
        };

        // Start animation
        animate();
      } catch (error) {
        console.error('Failed to initialize 3D viewer:', error);
        setError('Failed to initialize 3D viewer');
      }
    };

    init();

    // Cleanup function
    return () => {
      isActiveRef.current = false;
      
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
          if (object.geometry) {
            object.geometry.dispose();
          }
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
  }, []);

  // Handle play state
  useEffect(() => {
    if (!controlsRef.current) return;
    controlsRef.current.autoRotate = isPlaying;
    controlsRef.current.autoRotateSpeed = 2.0;
  }, [isPlaying]);

  // Handle window resize
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

  // Handle manifold updates
  useEffect(() => {
    if (!sceneRef.current || !manifoldData) {
      console.log('Missing scene or manifold data:', { scene: !!sceneRef.current, manifoldData });
      return;
    }
    console.log('Updating manifold:', manifoldData.type);

    // Clear existing objects
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
            opacity: 0.9, 
            shininess: 100, 
            side: THREE.DoubleSide,
            specular: 0x444444 
          });
          break;

        case 'torus':
          geometry = new THREE.TorusGeometry(3, 1, 30, 30);
           material = new THREE.MeshPhongMaterial({
             color: 0x156289,
             transparent: true,
             opacity: 0.9,
             shininess: 100,
             side: THREE.DoubleSide,
             specular: 0x444444
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
            opacity: 0.2, 
            linewidth: 1
          });
          const wireframeMesh = new THREE.LineSegments(wireframe, wireframeMaterial);
          wireframeMesh.name = 'manifold-wireframe';
          sceneRef.current.add(wireframeMesh);
        }

        // Reset camera and controls based on manifold type
        if (cameraRef.current && controlsRef.current) {
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
          controlsRef.current.target.set(0, 0, 0);
          controlsRef.current.update();
        }
      }
    } catch (error) {
      console.error('Error creating manifold:', error);
      setError('Failed to create manifold visualization');
    }
  }, [manifoldData]);

  // Handle trajectory updates
  useEffect(() => {
    if (!sceneRef.current || !trajectory?.length) return;

    const oldLine = sceneRef.current.getObjectByName('trajectory');
    const oldParticle = sceneRef.current.getObjectByName('particle');
    
    if (oldLine) {
      sceneRef.current.remove(oldLine);
      oldLine.geometry.dispose();
      oldLine.material.dispose();
    }
    
    if (oldParticle) {
      sceneRef.current.remove(oldParticle);
      oldParticle.geometry.dispose();
      oldParticle.material.dispose();
    }

    // Create trajectory line
    const points = trajectory.map(point => new THREE.Vector3(...point));
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0xff0000,
      linewidth: 2
    });
    const line = new THREE.Line(lineGeometry, lineMaterial);
    line.name = 'trajectory';
    sceneRef.current.add(line);

    // Create particle at current position
    if (points.length > 0) {
      const particleGeometry = new THREE.SphereGeometry(0.1, 16, 16);
      const particleMaterial = new THREE.MeshBasicMaterial({ 
        color: 0xff0000,
        opacity: 1,
        transparent: false
      });
      const particle = new THREE.Mesh(particleGeometry, particleMaterial);
      particle.position.copy(points[points.length - 1]);
      particle.name = 'particle';
      sceneRef.current.add(particle);
    }
  }, [trajectory]);

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