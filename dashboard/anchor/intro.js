// Lightweight Earth intro: a rotating globe + sparse starfield.
// Tuned for low-end machines: low-poly sphere, 2 meshes, few stars, capped DPR,
// and full GPU teardown before the avatar (PIXI) starts so the two WebGL
// contexts never run heavy at the same time.

import * as THREE from "./vendor/three.module.js";
import getStarfield from "./lib/getStarfield.js";

export function runEarthIntro({ container, spins = 2, duration = 3000 } = {}) {
  return new Promise((resolve) => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 4.2;

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "low-power" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.domElement.className = "intro-canvas";
    container.appendChild(renderer.domElement);

    const earthGroup = new THREE.Group();
    earthGroup.rotation.z = (-23.4 * Math.PI) / 180;
    scene.add(earthGroup);

    const loader = new THREE.TextureLoader();
    // Low-poly sphere: detail 6 (~7x fewer triangles than 14).
    const geometry = new THREE.IcosahedronGeometry(1, 6);
    const textures = [];

    const earthTex = loader.load("./images/earthmap.jpg");
    textures.push(earthTex);
    const earthMesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ map: earthTex }));
    earthGroup.add(earthMesh);

    // City lights add depth cheaply (additive, no extra geometry cost beyond a mesh).
    const lightsTex = loader.load("./images/earth_lights.png");
    textures.push(lightsTex);
    const lightsMesh = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ map: lightsTex, blending: THREE.AdditiveBlending, transparent: true })
    );
    earthGroup.add(lightsMesh);

    const stars = getStarfield({ numStars: 800 });
    scene.add(stars);

    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(-2, 0.5, 1.5);
    scene.add(sun);
    scene.add(new THREE.AmbientLight(0x335577, 0.7));

    let raf;
    const start = performance.now();
    const totalRotation = spins * Math.PI * 2;
    let finished = false;

    function animate(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      earthMesh.rotation.y = eased * totalRotation;
      lightsMesh.rotation.y = eased * totalRotation;
      camera.position.z = 4.2 - eased * 0.5;
      renderer.render(scene, camera);

      if (t >= 1) {
        finish();
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

    function finish() {
      if (finished) return;
      finished = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.domElement.classList.add("fade-out");
      // Free every GPU resource before the avatar's WebGL context spins up.
      setTimeout(() => {
        geometry.dispose();
        textures.forEach((tex) => tex.dispose());
        earthMesh.material.dispose();
        lightsMesh.material.dispose();
        if (stars.geometry) stars.geometry.dispose();
        if (stars.material) stars.material.dispose();
        renderer.dispose();
        renderer.forceContextLoss();
        renderer.domElement.remove();
        resolve();
      }, 700);
    }

    raf = requestAnimationFrame(animate);
  });
}
