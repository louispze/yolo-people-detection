# Construire l'APK Android (P2F Smart Home)

L'app web React/Vite est empaquetee en application Android native via
[Capacitor](https://capacitorjs.com/). Deux facons d'obtenir l'APK : via
GitHub Actions (aucune installation locale) ou en local avec Android Studio.

Identite de l'app (voir `capacitor.config.ts`) :

- **appId** : `com.p2f.smarthome`
- **appName** : `P2F Smart Home`
- **webDir** : `dist`
- HTTP cleartext autorise (`androidScheme: http`, `cleartext: true`,
  `allowMixedContent: true`) afin de charger les **flux MJPEG HTTP** du
  Raspberry Pi (cameras YOLO) depuis la WebView.

> Le dossier `android/` n'est **pas commite** dans le depot : il est genere
> automatiquement par `npx cap add android` lors du build.

---

## 1. Via GitHub Actions (recommande, zero install)

Le workflow se trouve dans `.github/workflows/build-apk.yml` (a la racine du
depot).

1. Pousse ton code sur la branche `main` ou `master` :

   ```bash
   git add .
   git commit -m "build apk"
   git push
   ```

   (ou declenche manuellement : onglet **Actions** -> **Build Android APK** ->
   **Run workflow**).

2. Ouvre l'onglet **Actions** du depot sur GitHub.
3. Clique sur le run **Build Android APK** le plus recent et attends qu'il
   passe au vert.
4. En bas de la page du run, section **Artifacts**, telecharge
   **`app-debug`**. Le zip contient `app-debug.apk`.
5. Transfere l'APK sur ton telephone Android et installe-le (autoriser les
   "sources inconnues" si demande).

Le workflow effectue, dans `app/mobile` :
`npm ci` (ou `npm install`) -> `npm run build` -> setup Java 17 + Android SDK
-> `npx cap add android` (si absent) -> `npx cap sync android` ->
`./gradlew assembleDebug` -> upload de l'artifact.

---

## 2. En local (avec Android Studio)

### Prerequis

- **Node.js 20+** et npm
- **JDK 17** (Temurin recommande)
- **Android Studio** (avec Android SDK + platform-tools). Definir la variable
  d'environnement `ANDROID_HOME` (ou `ANDROID_SDK_ROOT`) vers le SDK, p.ex. :
  - Windows : `C:\Users\<toi>\AppData\Local\Android\Sdk`
  - macOS : `~/Library/Android/sdk`
  - Linux : `~/Android/Sdk`

### Etapes

Depuis `app/mobile/` :

```bash
# 1. Installer les dependances
npm install

# 2. Construire l'app web (genere dist/)
npm run build

# 3. Ajouter la plateforme Android (uniquement la 1re fois)
npx cap add android

# 4. Synchroniser le web build + plugins dans le projet Android
npx cap sync android

# 5. Construire l'APK debug
cd android
# Windows :
gradlew.bat assembleDebug
# macOS / Linux :
./gradlew assembleDebug
```

> Raccourci : le `package.json` fournit `npm run cap:sync`
> (= `npm run build && cap sync android`) et `npm run cap:open`
> (= `cap open android`) pour ouvrir le projet dans Android Studio.

### Recuperer l'APK

L'APK genere se trouve ici :

```
app/mobile/android/app/build/outputs/apk/debug/app-debug.apk
```

Installe-le sur l'appareil :

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

---

## Notes

- Apres chaque modification du code web : `npm run build` puis
  `npx cap sync android` avant de reconstruire l'APK.
- Pour ouvrir le projet dans Android Studio (build via l'IDE, signature,
  build release...) : `npx cap open android`.
- L'APK debug n'est **pas signe pour la production** : il sert au test /
  developpement.
