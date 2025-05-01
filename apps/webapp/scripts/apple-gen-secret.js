require('dotenv').config();
const jwt = require('jsonwebtoken');
const fs = require('fs');

const appleId = process.env.APPLE_CLIENT_ID; //service ID
const keyId = process.env.APPLE_KEY_ID; // 10 chars, your generated apple key id
const teamId = process.env.APPLE_TEAM_ID;
const privateKey = fs.readFileSync('.secret.apple.p8');

const secret = jwt.sign(
  {
    iss: teamId,
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 86400 * 180, // 6 months
    aud: 'https://appleid.apple.com',
    sub: appleId,
  },
  privateKey,
  {
    algorithm: 'ES256',
    keyid: keyId,
  },
);

console.log(secret);
