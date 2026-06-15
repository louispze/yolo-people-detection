import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.p2f.smarthome",
  appName: "P2F Smart Home",
  webDir: "dist",
  server: {
    // Le Pi expose des flux MJPEG en HTTP simple (caméras YOLO).
    // On force le scheme http et on autorise le trafic non chiffre
    // pour pouvoir charger ces flux depuis la WebView Android.
    androidScheme: "http",
    cleartext: true,
  },
  android: {
    allowMixedContent: true,
  },
};

export default config;
