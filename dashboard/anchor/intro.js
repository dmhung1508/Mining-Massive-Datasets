// Earth intro: a rotating globe with clouds, glow and starfield.
// Spins for a few seconds, then resolves so the broadcast can begin.

import * as THREE from "./vendor/three.module.js";
import getStarfield from "./lib/getStarfield.js";
import { getFresnelMat } from "./lib/getFresnelMat.js";

export function runEarthIntro({ container, spins = 2.5, duration = 6500 } = {}) {
  return new Promise((resolve) => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 4.2;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.className = "intro-canvas";
    container.appendChild(renderer.domElement);

    THREE.ColorManagement.enabled = true;

    const earthGroup = new THREE.Group();
    earthGroup.rotation.z = (-23.4 * Math.PI) / 180;
    scene.add(earthGroup);

    const loader = new THREE.TextureLoader();
    const geometry = new THREE.IcosahedronGeometry(1, 14);

    const earthMesh = new THREE.Mesh(
      geometry,
      new THREE.MeshPhongMaterial({ map: loader.load("./images/earthmap.jpg") })
    );
    earthGroup.add(earthMesh);

    const lightsMesh = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        map: loader.load("./images/earth_lights.png"),
        blending: THREE.AdditiveBlending,
      })
    );
    earthGroup.add(lightsMesh);

    const cloudsMesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        map: loader.load("./images/cloud_combined.jpg"),
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
      })
    );
    cloudsMesh.scale.setScalar(1.003);
    earthGroup.add(cloudsMesh);

    const glowMesh = new THREE.Mesh(geometry, getFresnelMat({ rimHex: 0x2f6fff }));
    glowMesh.scale.setScalar(1.01);
    earthGroup.add(glowMesh);

    const stars = getStarfield({ numStars: 4000 });
    scene.add(stars);

    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(-2, 0.5, 1.5);
    scene.add(sun);
    scene.add(new THREE.AmbientLight(0x335577, 0.6));

    let raf;
    const start = performance.now();
    const totalRotation = spins * Math.PI * 2;

    function animate(now) {
      const t = Math.min(1, (now - start) / duration);
      // Ease-in-out so the spin accelerates then settles.
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      earthMesh.rotation.y = eased * totalRotation;
      lightsMesh.rotation.y = eased * totalRotation;
      cloudsMesh.rotation.y = eased * totalRotation * 1.1;
      glowMesh.rotation.y = eased * totalRotation;
      stars.rotation.y = -eased * 0.3;

      // Pull the camera in slightly toward the end for a "dive into Earth" feel.
      camera.position.z = 4.2 - eased * 0.6;

      renderer.render(scene, camera);

      if (t >= 1) {
        cancelAnimationFrame(raf);
        cleanup();
        resolve();
        return;
      }
      raf = requestAnimationFrame(animate);
    }

    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener("resize", onResize);

    function cleanup() {
      window.removeEventListener("resize", onResize);
      renderer.domElement.classList.add("fade-out");
      setTimeout(() => {
        renderer.domElement.remove();
        renderer.dispose();
        geometry.dispose();
      }, 900);
    }

    raf = requestAnimationFrame(animate);
  });
}
