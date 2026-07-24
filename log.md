# 2026-07-24T18:50:55    conversation: 01kyaqh9w9ffp99xe9wph2z1ws id: 01kyaqepne9n55jm4n7abvm185

Model: **anthropic/claude-opus-5**

## Prompt

Generate an SVG of a pelican riding a bicycle

## Response

Here's a pelican pedaling along on a bicycle:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="500" height="400">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#cfeeff"/>
      <stop offset="1" stop-color="#f6fcff"/>
    </linearGradient>
    <g id="wheel">
      <circle r="58" fill="none" stroke="#2b2b33" stroke-width="6"/>
      <circle r="50" fill="none" stroke="#8a8f99" stroke-width="2"/>
      <g stroke="#9aa1ab" stroke-width="2">
        <line x1="-50" y1="0" x2="50" y2="0"/>
        <line x1="0" y1="-50" x2="0" y2="50"/>
        <line x1="-35" y1="-35" x2="35" y2="35"/>
        <line x1="-35" y1="35" x2="35" y2="-35"/>
      </g>
      <circle r="6" fill="#2b2b33"/>
    </g>
  </defs>

  <!-- background -->
  <rect width="500" height="400" fill="url(#sky)"/>
  <circle cx="430" cy="70" r="26" fill="#fff3bf" opacity="0.9"/>
  <g fill="#ffffff" opacity="0.75">
    <ellipse cx="90" cy="80" rx="38" ry="16"/>
    <ellipse cx="115" cy="70" rx="26" ry="14"/>
  </g>
  <line x1="0" y1="361" x2="500" y2="361" stroke="#a9c48e" stroke-width="5"/>
  <g stroke="#9fb9d8" stroke-width="4" stroke-linecap="round" opacity="0.7">
    <line x1="40" y1="250" x2="95" y2="250"/>
    <line x1="20" y1="278" x2="80" y2="278"/>
    <line x1="45" y1="306" x2="90" y2="306"/>
  </g>

  <!-- ===== BICYCLE ===== -->
  <use href="#wheel" x="150" y="300"/>
  <use href="#wheel" x="352" y="300"/>

  <!-- chain -->
  <g stroke="#5b6068" stroke-width="3" fill="none">
    <line x1="150" y1="293" x2="250" y2="286"/>
    <line x1="150" y1="307" x2="250" y2="314"/>
  </g>

  <!-- frame -->
  <g stroke="#c1362f" stroke-width="8" stroke-linecap="round" fill="none">
    <line x1="150" y1="300" x2="250" y2="300"/>
    <line x1="150" y1="300" x2="214" y2="214"/>
    <line x1="250" y1="300" x2="214" y2="214"/>
    <line x1="214" y1="212" x2="330" y2="199"/>
    <line x1="250" y1="300" x2="344" y2="240"/>
    <line x1="330" y1="198" x2="344" y2="241"/>
    <line x1="344" y1="241" x2="352" y2="300"/>
  </g>

  <!-- handlebar -->
  <path d="M 330 200 C 330 180 340 172 358 175" stroke="#2b2b33" stroke-width="7"
        fill="none" stroke-linecap="round"/>

  <!-- seat post + saddle -->
  <line x1="214" y1="216" x2="214" y2="205" stroke="#2b2b33" stroke-width="7"/>
  <path d="M 186 212 C 198 202 232 202 240 211 C 228 221 198 221 186 212 Z" fill="#2b2b33"/>

  <!-- cranks and pedals -->
  <g stroke="#2b2b33" stroke-width="6" stroke-linecap="round">
    <line x1="250" y1="300" x2="228" y2="279"/>
    <line x1="250" y1="300" x2="272" y2="320"/>
    <line x1="216" y1="279" x2="240" y2="279"/>
    <line x1="260" y1="321" x2="284" y2="321"/>
  </g>
  <circle cx="250" cy="300" r="13" fill="none" stroke="#2b2b33" stroke-width="4"/>

  <!-- ===== PELICAN ===== -->
  <!-- back leg -->
  <path d="M 206 206 L 220 246 L 228 275" stroke="#d98736" stroke-width="9"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- tail feathers -->
  <path d="M 158 152 L 116 126 L 142 156 L 108 152 L 150 180 Z"
        fill="#e7edf4" stroke="#c4d0dd" stroke-width="2" stroke-linejoin="round"/>

  <!-- body -->
  <ellipse cx="205" cy="170" rx="58" ry="44" transform="rotate(-12 205 170)"
           fill="#fcfdff" stroke="#c4d0dd" stroke-width="2"/>

  <!-- front leg -->
  <path d="M 220 200 L 242 252 L 268 313" stroke="#eb9b45" stroke-width="9"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 256 312 L 286 312 L 282 325 L 258 323 Z" fill="#eb9b45"
        stroke="#d98736" stroke-width="2" stroke-linejoin="round"/>

  <!-- neck -->
  <path d="M 243 156 C 252 112 268 88 296 85" stroke="#fcfdff" stroke-width="33"
        fill="none" stroke-linecap="round"/>
  <path d="M 243 156 C 252 112 268 88 296 85" stroke="#dde6ef" stroke-width="33"
        fill="none" stroke-linecap="round" opacity="0.35"/>

  <!-- wing / arm reaching to handlebar -->
  <path d="M 250 168 C 288 160 320 168 344 180" stroke="#f0f5fa" stroke-width="19"
        fill="none" stroke-linecap="round"/>
  <path d="M 250 168 C 288 160 320 168 344 180" stroke="#c9d6e3" stroke-width="19"
        fill="none" stroke-linecap="round" opacity="0.25"/>

  <!-- wing -->
  <path d="M 219 142 C 252 137 271 160 264 191 C 252 215 210 217 189 199
           C 176 186 187 150 219 142 Z"
        fill="#eef3f9" stroke="#c4d0dd" stroke-width="2"/>
  <g stroke="#c9d6e3" stroke-width="2" fill="none">
    <path d="M 200 194 C 216 186 234 180 254 182"/>
    <path d="M 194 182 C 212 173 232 167 252 168"/>
  </g>

  <!-- head -->
  <circle cx="302" cy="84" r="25" fill="#fcfdff" stroke="#c4d0dd" stroke-width="2"/>
  <path d="M 296 60 C 306 52 318 56 320 64 C 312 60 302 60 296 60 Z" fill="#eef3f9"/>

  <!-- pouch -->
  <path d="M 320 90 C 330 134 378 136 400 107 Z" fill="#f0a755" stroke="#d98736" stroke-width="2"/>

  <!-- beak -->
  <path d="M 316 68 L 420 97 L 399 109 C 360 105 330 94 313 83 Z"
        fill="#f8bf6d" stroke="#dd9a3f" stroke-width="2" stroke-linejoin="round"/>
  <line x1="322" y1="76" x2="404" y2="99" stroke="#e0a749" stroke-width="2"/>

  <!-- eye -->
  <circle cx="306" cy="76" r="5" fill="#2b2b33"/>
  <circle cx="307.6" cy="74.4" r="1.7" fill="#ffffff"/>
</svg>
```

The pelican leans forward with one wingtip on the handlebars, webbed feet on the pedals, and that magnificent pouched beak leading the way.

## Token usage

25 input, 6,481 output

