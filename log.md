# 2026-07-24T23:05:06    conversation: 01kyb627ag7kn527xzgcrp1e2p id: 01kyb604tqaehzzqq584er57ek

Model: **anthropic/claude-opus-5**

## Prompt

Generate an SVG of a pelican riding a bicycle

## Response

Here's a pelican on a bicycle:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <!-- background -->
  <rect width="800" height="600" fill="#eaf4fb"/>
  <circle cx="660" cy="110" r="45" fill="#ffe9a8"/>
  <ellipse cx="120" cy="520" rx="130" ry="16" fill="#d4e6d2"/>
  <ellipse cx="700" cy="515" rx="110" ry="14" fill="#d4e6d2"/>
  <line x1="0" y1="530" x2="800" y2="530" stroke="#b9d3bb" stroke-width="6"/>

  <!-- shadows -->
  <ellipse cx="250" cy="532" rx="80" ry="10" fill="#000" opacity="0.12"/>
  <ellipse cx="550" cy="532" rx="80" ry="10" fill="#000" opacity="0.12"/>

  <!-- ============ BICYCLE ============ -->
  <g stroke="#2b3138" fill="none">
    <!-- rear wheel -->
    <g transform="translate(250,440)">
      <g stroke="#98a2ab" stroke-width="2">
        <line x1="0" y1="-78" x2="0" y2="78"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(30)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(60)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(90)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(120)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(150)"/>
      </g>
      <circle r="82" stroke-width="9"/>
      <circle r="8" fill="#2b3138" stroke-width="3"/>
    </g>
    <!-- front wheel -->
    <g transform="translate(550,440)">
      <g stroke="#98a2ab" stroke-width="2">
        <line x1="0" y1="-78" x2="0" y2="78"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(30)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(60)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(90)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(120)"/>
        <line x1="0" y1="-78" x2="0" y2="78" transform="rotate(150)"/>
      </g>
      <circle r="82" stroke-width="9"/>
      <circle r="8" fill="#2b3138" stroke-width="3"/>
    </g>

    <!-- chain -->
    <g stroke="#6c7680" stroke-width="3">
      <path d="M400 418 L252 431"/>
      <path d="M400 462 L252 449"/>
    </g>

    <!-- frame -->
    <g stroke="#d94f3d" stroke-width="10" stroke-linecap="round">
      <path d="M400 440 L250 440"/>      <!-- chainstay -->
      <path d="M400 440 L348 332"/>      <!-- seat tube -->
      <path d="M348 332 L250 440"/>      <!-- seat stay -->
      <path d="M400 440 L482 342"/>      <!-- down tube -->
      <path d="M348 332 L482 342"/>      <!-- top tube -->
      <path d="M482 342 L502 302"/>      <!-- head tube -->
      <path d="M482 342 L550 440"/>      <!-- fork -->
    </g>

    <!-- crank + pedals -->
    <circle cx="400" cy="440" r="21" stroke="#6c7680" stroke-width="4"/>
    <g stroke="#3a4149" stroke-width="7" stroke-linecap="round">
      <path d="M400 440 L378 462"/>
      <path d="M400 440 L422 418"/>
    </g>
    <g stroke="#2b3138" stroke-width="6" stroke-linecap="round">
      <path d="M366 466 L392 458"/>
      <path d="M410 422 L436 414"/>
    </g>

    <!-- seat -->
    <path d="M316 326 Q345 312 372 322 Q356 336 322 334 Z" fill="#2b3138" stroke-width="3"/>
    <path d="M348 332 L352 344" stroke="#6c7680" stroke-width="7"/>

    <!-- handlebar -->
    <g stroke="#2b3138" stroke-width="8" stroke-linecap="round">
      <path d="M502 302 L524 293 Q552 288 553 308 Q553 326 536 330"/>
    </g>
  </g>

  <!-- ============ PELICAN ============ -->
  <!-- tail feathers -->
  <g fill="#e7ebef" stroke="#b9c3cc" stroke-width="2">
    <path d="M330 250 L268 214 L318 262 Z"/>
    <path d="M328 268 L262 252 L316 284 Z"/>
  </g>

  <!-- legs -->
  <g stroke="#f0952f" stroke-width="11" stroke-linecap="round" fill="none">
    <path d="M372 306 Q392 372 380 452"/>
    <path d="M398 300 Q426 356 424 410"/>
  </g>
  <g fill="#f0952f">
    <path d="M380 452 l-20 8 l4 10 l30 -4 z"/>
    <path d="M424 410 l-20 8 l4 10 l30 -4 z"/>
  </g>

  <!-- body -->
  <ellipse cx="382" cy="258" rx="88" ry="64" transform="rotate(-20 382 258)" fill="#fbfdff" stroke="#c3ced8" stroke-width="3"/>
  <path d="M320 300 Q385 336 450 280 Q400 322 322 302 Z" fill="#e3eaf1"/>

  <!-- neck -->
  <path d="M416 244 C 434 200, 460 172, 486 162" fill="none" stroke="#fbfdff" stroke-width="40" stroke-linecap="round"/>
  <path d="M416 244 C 434 200, 460 172, 486 162" fill="none" stroke="#c3ced8" stroke-width="40" stroke-linecap="round" opacity="0.25"/>
  <path d="M420 246 C 438 204, 462 178, 488 168" fill="none" stroke="#fbfdff" stroke-width="30" stroke-linecap="round"/>

  <!-- head -->
  <circle cx="500" cy="160" r="33" fill="#fbfdff" stroke="#c3ced8" stroke-width="3"/>
  <path d="M480 133 Q498 118 516 132 Q500 128 480 138 Z" fill="#e3eaf1"/>

  <!-- pouch -->
  <path d="M523 168 C 528 232, 600 228, 648 182 L 640 176 C 596 208, 542 208, 530 162 Z" fill="#f7b93f" stroke="#d9922a" stroke-width="3"/>
  <!-- beak -->
  <path d="M521 143 Q580 143 652 174 L 648 184 Q582 164 520 166 Z" fill="#ffcf5c" stroke="#d9922a" stroke-width="3"/>
  <path d="M527 152 Q585 154 640 176" fill="none" stroke="#d9922a" stroke-width="2"/>

  <!-- eye -->
  <circle cx="508" cy="150" r="6.5" fill="#26313a"/>
  <circle cx="510" cy="148" r="2.2" fill="#fff"/>

  <!-- folded wing on body -->
  <path d="M340 224 C 398 202, 452 236, 442 288 C 432 330, 360 330, 332 288 Z" fill="#eef3f8" stroke="#b9c3cc" stroke-width="3"/>
  <g stroke="#c3ced8" stroke-width="2.5" fill="none">
    <path d="M356 300 Q392 288 420 258"/>
    <path d="M348 284 Q386 272 414 244"/>
    <path d="M344 266 Q378 256 402 234"/>
  </g>

  <!-- wing reaching handlebar -->
  <path d="M424 236 C 466 246, 500 272, 520 296 L 500 314 C 476 288, 444 268, 414 262 Z" fill="#fbfdff" stroke="#c3ced8" stroke-width="3"/>
  <ellipse cx="512" cy="306" rx="16" ry="13" transform="rotate(-30 512 306)" fill="#f7b93f" stroke="#d9922a" stroke-width="3"/>
</svg>
```

The pelican is perched on the saddle with one wing gripping the drop handlebar, its big orange pouch-beak leading the way, and webbed feet on the pedals.

## Token usage

25 input, 5,315 output

