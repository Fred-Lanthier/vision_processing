const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

(async () => {
  console.log('🚀 Démarrage des tunnels...');

  // 1. Tunnel Backend (8000)
  const backendTunnel = await localtunnel({ port: 8000 });
  const backendUrl = backendTunnel.url;
  console.log(`✅ Backend exposé sur : ${backendUrl}`);

  // 2. Mise à jour de la config Frontend
  const apiPath = path.join(__dirname, 'frontend/src/services/api.ts');
  let apiContent = fs.readFileSync(apiPath, 'utf8');
  
  // Remplace l'ancienne URL (quelle qu'elle soit) par la nouvelle
  const newContent = apiContent.replace(
    /const API_URL = '.*';/,
    `const API_URL = '${backendUrl}';`
  );
  
  fs.writeFileSync(apiPath, newContent);
  console.log('🔄 Frontend configuré avec la nouvelle URL API.');

  // 3. Tunnel Frontend (5173)
  // On attend un peu que Vite redémarre si besoin (mais le tunnel est indépendant)
  const frontendTunnel = await localtunnel({ port: 5173 });
  const frontendUrl = frontendTunnel.url;

  console.log('\n' + '='.repeat(50));
  console.log('📱 SUCCÈS ! OUVREZ CETTE URL SUR VOTRE TÉLÉPHONE :');
  console.log('\n   👉 ' + frontendUrl + '\n');
  console.log('='.repeat(50));
  console.log('\n⚠️  Gardez ce terminal ouvert.');
  console.log('⚠️  Assurez-vous que vos serveurs Backend et Frontend tournent dans d\'autres terminaux !');

  // Gestion de la fermeture
  backendTunnel.on('close', () => {
    console.log('Backend tunnel closed');
  });
  frontendTunnel.on('close', () => {
    console.log('Frontend tunnel closed');
  });

})();
