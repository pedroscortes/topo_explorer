import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const GeometryView = ({ manifoldData, trajectory, isPlaying }) => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const [error, setError] = useState(null);

  // Scene setup
  useEffect(() => {
    if (!containerRef.current) return;

    try {
      // Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf5f5f5);
      sceneRef.current = scene;

      // Camera
      const camera = new THREE.PerspectiveCamera(
        75,
        containerRef.current.clientWidth / containerRef.current.clientHeight,
        0.1,
        1000
      );
      camera.position.set(5, 5, 5);
      camera.lookAt(0, 0, 0);
      cameraRef.current = camera;

      // Renderer
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
      });
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      containerRef.current.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 3;
      controls.maxDistance = 20;
      controlsRef.current = controls;

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 5, 5);
      scene.add(directionalLight);

      // Grid helper and axes
      const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
      gridHelper.material.opacity = 0.2;
      gridHelper.material.transparent = true;
      scene.add(gridHelper);

      const axesHelper = new THREE.AxesHelper(3);
      scene.add(axesHelper);

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        renderer.render(scene, camera);
      };
      animate();

      // Handle window resize
      const handleResize = () => {
        if (!containerRef.current) return;
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        renderer.dispose();
        scene.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            object.material.dispose();
          }
        });
        containerRef.current?.removeChild(renderer.domElement);
      };
    } catch (error) {
      console.error('Failed to initialize 3D viewer:', error);
      setError('Failed to initialize 3D viewer');
    }
  }, []);

  // Update manifold visualization
  useEffect(() => {
    if (!sceneRef.current || !manifoldData) return;

    const scene = sceneRef.current;
    
    // Remove existing manifold and its children
    const existingManifold = scene.getObjectByName('manifold');
    if (existingManifold) {
      existingManifold.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      scene.remove(existingManifold);
    }

    console.log('Creating manifold:', manifoldData.type);

    // Create new manifold based on type
    let geometry;
    const radius = manifoldData.radius || 2;

    switch (manifoldData.type) {
      case 'sphere':
        geometry = new THREE.SphereGeometry(radius, 64, 32);
        break;
      case 'torus':
        geometry = new THREE.TorusGeometry(3, 1, 48, 64);
        break;
      case 'mobius':
        // Create Möbius strip
        geometry = new THREE.ParametricGeometry((u, v, target) => {
          u = u * Math.PI * 2;
          v = v * 2 - 1;
          const r = 3;
          const halfWidth = 1;
          target.x = (r + halfWidth * v * Math.cos(u/2)) * Math.cos(u);
          target.y = (r + halfWidth * v * Math.cos(u/2)) * Math.sin(u);
          target.z = halfWidth * v * Math.sin(u/2);
        }, 100, 20);
        break;
      case 'klein':
        // Create Klein bottle approximation
        geometry = new THREE.ParametricGeometry((u, v, target) => {
          u = u * Math.PI * 2;
          v = v * Math.PI * 2;
          const a = 3;
          const n = 4;
          target.x = (a + Math.cos(u/2) * Math.sin(v) - Math.sin(u/2) * Math.sin(2*v)) * Math.cos(u);
          target.y = (a + Math.cos(u/2) * Math.sin(v) - Math.sin(u/2) * Math.sin(2*v)) * Math.sin(u);
          target.z = Math.sin(u/2) * Math.sin(v) + Math.cos(u/2) * Math.sin(2*v);
        }, 100, 20);
        break;
      case 'hyperbolic':
        // Create hyperbolic paraboloid
        geometry = new THREE.ParametricGeometry((u, v, target) => {
          u = (u - 0.5) * 4;
          v = (v - 0.5) * 4;
          target.x = u;
          target.y = v;
          target.z = (u*u - v*v) / 4;
        }, 50, 50);
        break;
      case 'projective':
        // Create Boy's surface (projective plane)
        geometry = new THREE.ParametricGeometry((u, v, target) => {
          u = u * Math.PI * 2;
          v = v * Math.PI;
          const r = 2;
          const x = r * Math.cos(u) * Math.sin(v);
          const y = r * Math.sin(u) * Math.sin(v);
          const z = r * Math.cos(v);
          target.x = x;
          target.y = y;
          target.z = z;
        }, 100, 50);
        break;
      default:
        geometry = new THREE.SphereGeometry(radius, 64, 32);
    }

    // Create materials
    const material = new THREE.MeshPhongMaterial({
      color: 0x156289,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
      wireframe: false,
      flatShading: manifoldData.type === 'klein' || manifoldData.type === 'mobius'
    });

    const manifold = new THREE.Mesh(geometry, material);
    manifold.name = 'manifold';
    
    // Add wireframe
    const wireframe = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({
      color: 0x000000,
      opacity: 0.1,
      transparent: true
    });
    const wireframeMesh = new THREE.LineSegments(wireframe, wireframeMaterial);
    manifold.add(wireframeMesh);
    
    scene.add(manifold);

    // Adjust camera position based on manifold type
    if (cameraRef.current) {
      switch (manifoldData.type) {
        case 'hyperbolic':
          cameraRef.current.position.set(0, 0, 10);
          break;
        case 'torus':
          cameraRef.current.position.set(8, 8, 8);
          break;
        case 'klein':
        case 'mobius':
          cameraRef.current.position.set(7, 7, 7);
          break;
        default:
          cameraRef.current.position.set(5, 5, 5);
      }
      cameraRef.current.lookAt(0, 0, 0);
      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }
    }
  }, [manifoldData]);

  // Update trajectory and agent visualization
  useEffect(() => {
    if (!sceneRef.current || !trajectory?.length) return;

    const scene = sceneRef.current;
    
    // Remove existing trajectory
    const existingTrajectory = scene.getObjectByName('trajectory');
    if (existingTrajectory) {
      scene.remove(existingTrajectory);
      existingTrajectory.geometry.dispose();
      existingTrajectory.material.dispose();
    }

    // Remove existing agent
    const existingAgent = scene.getObjectByName('agent');
    if (existingAgent) {
      scene.remove(existingAgent);
      existingAgent.geometry.dispose();
      existingAgent.material.dispose();
    }

    // Create trajectory line
    const points = trajectory.map(point => new THREE.Vector3(...point));
    const trajectoryGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const trajectoryMaterial = new THREE.LineBasicMaterial({
      color: 0xff0000,
      linewidth: 2,
    });
    const trajectoryLine = new THREE.Line(trajectoryGeometry, trajectoryMaterial);
    trajectoryLine.name = 'trajectory';
    scene.add(trajectoryLine);

    // Create agent (particle)
    if (points.length > 0) {
      const agentGeometry = new THREE.SphereGeometry(0.1, 16, 16);
      const agentMaterial = new THREE.MeshPhongMaterial({
        color: 0xff0000,
        emissive: 0xff0000,
        emissiveIntensity: 0.5,
      });
      const agent = new THREE.Mesh(agentGeometry, agentMaterial);
      agent.position.copy(points[points.length - 1]);
      agent.name = 'agent';
      scene.add(agent);

      // Add point light to agent
      const pointLight = new THREE.PointLight(0xff0000, 1, 2);
      pointLight.position.copy(agent.position);
      agent.add(pointLight);
    }
  }, [trajectory]);

  // Handle play/pause state
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isPlaying;
      controlsRef.current.autoRotateSpeed = 1.0;
    }
  }, [isPlaying]);

  return (
    <div className="w-full h-96 relative bg-gray-100">
      {error ? (
        <div className="w-full h-full flex items-center justify-center text-red-600">
          {error}
        </div>
      ) : (
        <div ref={containerRef} className="w-full h-full" />
      )}
    </div>
  );
};

export default GeometryView;