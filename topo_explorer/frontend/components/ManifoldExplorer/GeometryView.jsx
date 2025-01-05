import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const GeometryView = ({ manifoldData, trajectory, isPlaying }) => {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    sceneRef.current = new THREE.Scene();
    sceneRef.current.background = new THREE.Color(0xf0f0f0);

    const aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
    cameraRef.current = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    cameraRef.current.position.set(3, 3, 3);
    cameraRef.current.lookAt(0, 0, 0);

    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true
    });
    rendererRef.current.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    rendererRef.current.setPixelRatio(window.devicePixelRatio);

    controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.05;
    controlsRef.current.rotateSpeed = 0.5;
    controlsRef.current.enablePan = true;
    controlsRef.current.enableZoom = true;
    controlsRef.current.minDistance = 2;
    controlsRef.current.maxDistance = 10;

    const ambientLight = new THREE.AmbientLight(0x404040);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    sceneRef.current.add(ambientLight);
    sceneRef.current.add(directionalLight);

    const axesHelper = new THREE.AxesHelper(2);
    sceneRef.current.add(axesHelper);

    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: 0x156289,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
    });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.name = 'manifold';
    sceneRef.current.add(sphere);

    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };
    animate();

    const handleResize = () => {
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (rendererRef.current && sceneRef.current) {
        sceneRef.current.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            object.material.dispose();
          }
        });
        rendererRef.current.dispose();
        controlsRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current || !trajectory?.length) return;

    const oldLine = sceneRef.current.getObjectByName('trajectory');
    if (oldLine) {
      sceneRef.current.remove(oldLine);
      oldLine.geometry.dispose();
      oldLine.material.dispose();
    }

    const points = trajectory.map(point => new THREE.Vector3(...point));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xff0000,
      linewidth: 2
    });
    const line = new THREE.Line(geometry, material);
    line.name = 'trajectory';
    sceneRef.current.add(line);

    const currentPos = points[points.length - 1];
    if (currentPos) {
      const markerGeometry = new THREE.SphereGeometry(0.05, 16, 16);
      const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(currentPos);
      marker.name = 'currentPos';
      
      const oldMarker = sceneRef.current.getObjectByName('currentPos');
      if (oldMarker) {
        sceneRef.current.remove(oldMarker);
        oldMarker.geometry.dispose();
        oldMarker.material.dispose();
      }
      sceneRef.current.add(marker);
    }
  }, [trajectory]);

  useEffect(() => {
    if (!sceneRef.current || !manifoldData) return;

    const oldManifold = sceneRef.current.getObjectByName('manifold');
    if (oldManifold) {
      sceneRef.current.remove(oldManifold);
      oldManifold.geometry.dispose();
      oldManifold.material.dispose();
    }

    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: 0x156289,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
    });
    const manifold = new THREE.Mesh(geometry, material);
    manifold.name = 'manifold';
    sceneRef.current.add(manifold);
  }, [manifoldData]);

  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isPlaying;
    }
  }, [isPlaying]);

  return (
    <canvas 
      ref={canvasRef}
      className="w-full h-96"
    />
  );
};

export default GeometryView;