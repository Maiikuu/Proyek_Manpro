// SPDX-License-Identifier: MIT
// Copyright contributors to the kepler.gl project

import React from 'react';

const LINK_STYLE = {display: 'flex'};

const NetlifyLogo = () => (
  <a href="https://www.netlify.com" style={LINK_STYLE}>
    <svg width="114" height="51">
      <g
        xmlns="http://www.w3.org/2000/svg"
        stroke="none"
        strokeWidth="1"
        fill="none"
        fillRule="evenodd"
      >
        <g>
          <g transform="translate(8.000000, 8.000000)" fill="#FFFFFF">
            <path
              d="M23.6161838,12.0243902 C23.7871091,12.1101766 23.9295469,12.2245585 24.0434971,12.353238 C24.0577409,12.3675357 24.0577409,12.3675357 24.0719847,12.3675357 L24.0862285,12.3675357 L27.3622974,10.9520606 C27.3765412,10.9377628 27.3907849,10.9234651 27.3907849,10.9091674 C27.3907849,10.8948696 27.3907849,10.8805719 27.3765412,10.8662742 L24.3141289,7.79226241 C24.2998851,7.77796468 24.2856414,7.77796468 24.2856414,7.77796468 L24.2713976,7.77796468 C24.2571538,7.77796468 24.24291,7.79226241 24.24291,7.82085786 L23.5734525,11.9814971 C23.5876962,11.9957948 23.60194,12.0243902 23.6161838,12.0243902 Z M16.8219017,9.23633305 C16.9785833,9.47939445 17.0782897,9.76534903 17.1067773,10.0513036 C17.1067773,10.0656013 17.1210211,10.0798991 17.1352648,10.0941968 L22.0066369,12.195963 L22.0208807,12.195963 C22.0351244,12.195963 22.0493682,12.195963 22.0493682,12.1816653 C22.191806,12.0672834 22.3627313,11.9814971 22.5479004,11.9243061 C22.5621442,11.9243061 22.576388,11.9100084 22.576388,11.881413 L23.3740396,6.86291001 C23.3740396,6.84861228 23.3740396,6.83431455 23.3597958,6.82001682 L20.3116273,3.74600505 C20.2973835,3.73170732 20.2973835,3.73170732 20.2831398,3.73170732 C20.268896,3.73170732 20.2546522,3.74600505 20.2546522,3.76030278 L16.8219017,9.16484441 C16.8076579,9.19343987 16.8076579,9.22203532 16.8219017,9.23633305 Z M33.4586343,16.9714045 L28.2311678,11.7098402 C28.2169241,11.6955425 28.2026803,11.6955425 28.2026803,11.6955425 L28.1884365,11.6955425 L24.6417358,13.2253995 C24.627492,13.2396972 24.6132482,13.253995 24.6132482,13.2682927 C24.6132482,13.2825904 24.627492,13.3111859 24.6417358,13.3111859 L33.3874154,17.0714886 L33.4016592,17.0714886 C33.415903,17.0714886 33.4301468,17.0714886 33.4301468,17.0571909 L33.4586343,17.0285955 C33.4871219,17.0285955 33.4871219,16.9857023 33.4586343,16.9714045 Z M32.5897639,17.8292683 L24.2001787,14.2262405 L24.1859349,14.2262405 C24.1716911,14.2262405 24.1574474,14.2262405 24.1432036,14.2405383 C23.9153031,14.5550883 23.5734525,14.7552565 23.1746267,14.8124474 C23.1603829,14.8124474 23.1318953,14.8267452 23.1318953,14.8553406 L22.2345373,20.4457527 C22.2345373,20.4600505 22.2345373,20.4743482 22.2487811,20.4886459 C22.5621442,20.7317073 22.7473133,21.0891505 22.7900447,21.489487 C22.7900447,21.5180824 22.8042884,21.5323802 22.832776,21.5323802 L27.9035609,22.6047098 L27.9178047,22.6047098 C27.9320485,22.6047098 27.9462923,22.6047098 27.9462923,22.5904121 L32.5897639,17.9150547 C32.6040077,17.9007569 32.6040077,17.8864592 32.6040077,17.8721615 C32.6040077,17.8578638 32.6040077,17.843566 32.5897639,17.8292683 Z M21.4796171,13.0538267 L16.8931206,11.0807401 L16.8788768,11.0807401 C16.8646331,11.0807401 16.8503893,11.0950378 16.8361455,11.1093356 C16.5227824,11.5954584 15.9957626,11.881413 15.4260115,11.881413 C15.3405488,11.881413 15.2550862,11.8671152 15.1553797,11.8528175 L15.1411359,11.8528175 C15.1268922,11.8528175 15.1126484,11.8671152 15.0984046,11.881413 L11.3238034,17.8149706 C11.3095597,17.8292683 11.3095597,17.8578638 11.3238034,17.8721615 C11.3380472,17.8864592 11.352291,17.8864592 11.3665348,17.8864592 L11.3807786,17.8864592 L21.4511295,13.5256518 C21.4653733,13.5113541 21.4796171,13.4970563 21.4796171,13.4827586 L21.4796171,13.4255677 L21.4796171,13.3540791 C21.4796171,13.2682927 21.4938609,13.1825063 21.5081047,13.1110177 C21.5081047,13.0824222 21.4938609,13.0681245 21.4796171,13.0538267 Z M27.0062029,23.4339781 L22.5479004,22.5046257 L22.5336567,22.5046257 C22.5194129,22.5046257 22.5051691,22.5189235 22.4909253,22.5189235 C22.32,22.7333894 22.1063433,22.9049622 21.8499553,23.0050463 C21.8357116,23.0050463 21.8214678,23.0336417 21.8214678,23.0479394 L20.7531844,29.7106812 C20.7531844,29.7392767 20.7674282,29.7535744 20.781672,29.7678722 L20.8101595,29.7678722 C20.8244033,29.7678722 20.8386471,29.7678722 20.8386471,29.7535744 L27.0204467,23.5340622 C27.0346905,23.5197645 27.0346905,23.5054668 27.0346905,23.491169 C27.0346905,23.4482759 27.0204467,23.4339781 27.0062029,23.4339781 Z M20.781672,22.9764508 C20.3543586,22.804878 20.0409955,22.4188394 19.9270453,21.9756098 C19.9270453,21.961312 19.9128015,21.9470143 19.884314,21.9327166 L11.6229228,20.2026913 C11.6229228,20.2026913 11.6229228,20.2026913 11.608679,20.2026913 C11.5944352,20.2026913 11.5801914,20.2169891 11.5659477,20.2312868 C11.5232163,20.3027754 11.4947288,20.3599664 11.4519974,20.4171573 C11.4377537,20.431455 11.4377537,20.4600505 11.4519974,20.4743482 L18.9727122,31.5121951 C18.986956,31.5264929 18.986956,31.5264929 19.0011997,31.5264929 C19.0154435,31.5264929 19.0296873,31.5264929 19.0296873,31.5121951 L19.4854882,31.0546678 C19.4854882,31.0403701 19.499732,31.0403701 19.499732,31.0260723 L20.781672,23.019344 C20.8101595,23.019344 20.8101595,22.9907485 20.781672,22.9764508 Z M11.7938481,19.1875526 C11.7938481,19.216148 11.8080919,19.2304458 11.8365795,19.2304458 L20.0267518,20.9461733 L20.0409955,20.9461733 C20.0552393,20.9461733 20.0694831,20.9318755 20.0837269,20.9175778 C20.3258711,20.4886459 20.7531844,20.2026913 21.2374729,20.1740959 C21.2659604,20.1740959 21.2802042,20.1597981 21.2802042,20.1312027 L22.1633184,14.626577 C22.1633184,14.6122792 22.1633184,14.5836838 22.1348309,14.5836838 C22.0778558,14.5407906 22.0208807,14.4978974 21.9496618,14.4264087 C21.935418,14.412111 21.9211742,14.412111 21.9211742,14.412111 L21.9069304,14.412111 L11.7796043,18.8015139 C11.7511168,18.8158116 11.7511168,18.8301093 11.7511168,18.8587048 C11.7653606,18.9730866 11.7938481,19.0731707 11.7938481,19.1875526 Z M8.36109764,20.5744323 C8.31836631,20.5172414 8.27563497,20.4600505 8.23290364,20.3885618 C8.21865986,20.3742641 8.20441608,20.3599664 8.1901723,20.3599664 L8.17592853,20.3599664 L4.6434716,21.8898234 C4.62922782,21.8898234 4.61498405,21.9041211 4.61498405,21.9184188 C4.61498405,21.9327166 4.61498405,21.9470143 4.62922782,21.961312 L6.35272495,23.6913373 C6.36696873,23.705635 6.38121251,23.705635 6.38121251,23.705635 C6.39545629,23.705635 6.40970006,23.6913373 6.42394384,23.6770395 L8.37534142,20.6030278 C8.37534142,20.6030278 8.37534142,20.58873 8.36109764,20.5744323 Z M10.6970772,21.1320437 C10.6828334,21.117746 10.6685897,21.1034483 10.6543459,21.1034483 L10.6401021,21.1034483 C10.3837141,21.2178301 10.1273261,21.275021 9.85669432,21.275021 C9.64303765,21.275021 9.44362476,21.2464256 9.22996809,21.1749369 L9.21572431,21.1749369 C9.20148054,21.1749369 9.18723676,21.1892347 9.17299298,21.2035324 L7.12188896,24.4348192 L7.10764518,24.4491169 C7.0934014,24.4634146 7.0934014,24.4920101 7.10764518,24.5063078 L16.5370262,33.9857023 C16.5512699,34 16.5655137,34 16.5655137,34 C16.5797575,34 16.5940013,34 16.5940013,33.9857023 L18.2462795,32.312868 C18.2605233,32.2985702 18.2605233,32.2699748 18.2462795,32.255677 L10.6970772,21.1320437 Z M9.37240587,17.4003364 C9.38664965,17.4146341 9.40089343,17.4289319 9.4151372,17.4289319 L9.42938098,17.4289319 C9.57181876,17.4003364 9.72850032,17.371741 9.8709381,17.371741 C10.0276197,17.371741 10.198545,17.4003364 10.3552265,17.4432296 L10.3694703,17.4432296 C10.3837141,17.4432296 10.3979579,17.4289319 10.4122017,17.4146341 L14.2295341,11.4095879 C14.2437779,11.3952902 14.2437779,11.3666947 14.2295341,11.352397 C13.9304148,11.0378469 13.7594895,10.6232128 13.7594895,10.1799832 C13.7594895,10.0513036 13.7737332,9.92262405 13.8022208,9.79394449 C13.8022208,9.76534903 13.787977,9.7510513 13.7737332,9.73675357 C13.2894448,9.52228764 9.00206765,7.6921783 9.00206765,7.67788057 L8.98782387,7.67788057 C8.97358009,7.67788057 8.95933631,7.67788057 8.95933631,7.6921783 L5.32717294,11.352397 C5.31292916,11.3666947 5.31292916,11.3952902 5.32717294,11.4095879 L9.37240587,17.4003364 Z M9.78547543,6.93439865 C9.78547543,6.93439865 14.1155839,8.79310345 14.300753,8.87888982 L14.3149968,8.87888982 C14.3292406,8.87888982 14.3292406,8.87888982 14.3434844,8.86459209 C14.6426037,8.6215307 15.0271857,8.47855341 15.4117677,8.47855341 C15.5969368,8.47855341 15.7821059,8.50714886 15.967275,8.56433978 L15.9815188,8.56433978 C15.9957626,8.56433978 16.0100064,8.55004205 16.0242502,8.53574432 L19.5424633,3.00252313 C19.5567071,2.9882254 19.5567071,2.95962994 19.5424633,2.94533221 L16.6224888,0.0142977292 C16.6082451,0 16.6082451,0 16.5940013,0 C16.5797575,0 16.5655137,0 16.5655137,0.0142977292 L9.78547543,6.84861228 C9.77123165,6.86291001 9.77123165,6.87720774 9.77123165,6.89150547 C9.75698787,6.92010093 9.77123165,6.92010093 9.78547543,6.93439865 Z M8.10470964,18.4440706 C8.11895341,18.4440706 8.13319719,18.4297729 8.14744097,18.4154752 C8.23290364,18.2439024 8.36109764,18.0866274 8.48929164,17.9436501 C8.50353542,17.9293524 8.50353542,17.9007569 8.48929164,17.8864592 C8.44656031,17.8292683 4.58649649,12.1673675 4.58649649,12.1530698 C4.57225271,12.1387721 4.57225271,12.1387721 4.54376516,12.1244743 C4.52952138,12.1244743 4.5152776,12.1244743 4.5152776,12.1387721 L0.0142437779,16.6568545 C0,16.6711522 0,16.68545 0,16.6997477 C0,16.7140454 0.0142437779,16.7283431 0.0427313338,16.7283431 L8.10470964,18.4440706 C8.09046586,18.4440706 8.09046586,18.4440706 8.10470964,18.4440706 Z M7.73437141,19.430614 C7.73437141,19.4020185 7.72012763,19.3877208 7.69164008,19.3877208 L0.697945118,17.9150547 C0.697945118,17.9150547 0.697945118,17.9150547 0.68370134,17.9150547 C0.669457562,17.9150547 0.655213784,17.9293524 0.640970006,17.9436501 C0.626726228,17.9579479 0.640970006,17.9865433 0.655213784,18.000841 L3.77460115,21.1463415 C3.78884493,21.1606392 3.8030887,21.1606392 3.8030887,21.1606392 L3.81733248,21.1606392 L7.69164008,19.4878049 C7.72012763,19.4592094 7.73437141,19.4449117 7.73437141,19.430614 Z"
              id="Combined-Shape-Copy"
            />
            <path
              d="M67.5018666,14.7649667 L69.7772159,14.7649667 L69.7772159,29.3120637 L67.5018666,29.3120637 L67.5018666,14.7649667 Z M44.2221988,18.869426 C42.9991985,18.869426 42.0179541,19.3301306 41.2642447,20.237579 L41.19314,19.0509157 L39.06,19.0509157 L39.06,29.298103 L41.3353493,29.298103 L41.3353493,22.0105937 C41.7904192,21.172949 42.473024,20.7541266 43.3831637,20.7541266 C44.0088848,20.7541266 44.4639547,20.9076948 44.7483733,21.228792 C45.032792,21.5359284 45.1607804,22.0245545 45.1607804,22.6667488 L45.1607804,29.298103 L47.4361297,29.298103 L47.4361297,22.5271413 C47.4076878,20.0979716 46.3411178,18.869426 44.2221988,18.869426 Z M54.0488637,18.869426 C53.1813868,18.869426 52.3992354,19.0927979 51.6881888,19.5395418 C50.9771421,19.9862856 50.4367466,20.6145192 50.0385605,21.4242424 C49.6545953,22.2339657 49.4555022,23.1414141 49.4555022,24.1605486 L49.4555022,24.4397635 C49.4555022,25.9614848 49.9105721,27.1900304 50.8064909,28.1114396 C51.7024097,29.0328488 52.8685262,29.4935534 54.3190614,29.4935534 C55.1580965,29.4935534 55.9260269,29.3260245 56.6086317,28.9909666 C57.2912365,28.6559087 57.831632,28.1812433 58.2298181,27.5669705 L57.0068178,26.3803071 C56.3526549,27.2319126 55.4993989,27.6646957 54.4612708,27.6646957 C53.7217822,27.6646957 53.0960612,27.4134023 52.6125494,26.9247762 C52.1148168,26.4361501 51.844619,25.7660343 51.7735144,24.9144288 L58.4004693,24.9144288 L58.4004693,23.9930196 C58.4004693,22.3596124 58.0165041,21.1031453 57.2770156,20.2096576 C56.4948642,19.3161698 55.4282942,18.869426 54.0488637,18.869426 Z M56.12512,23.2251786 L51.7877353,23.2251786 C51.8872818,22.4294161 52.1432586,21.8151433 52.5272238,21.3823602 C52.911189,20.9356163 53.4231426,20.7262051 54.0488637,20.7262051 C54.6745848,20.7262051 55.1723174,20.9216556 55.5278408,21.3125565 C55.8833641,21.7034573 56.0824572,22.2898087 56.1393409,23.0576497 L56.1393409,23.2251786 L56.12512,23.2251786 Z M63.5200053,27.3296378 C63.363575,27.1621089 63.2924703,26.8689332 63.2924703,26.4780324 L63.2924703,20.7541266 L65.0843079,20.7541266 L65.0843079,19.0509157 L63.2924703,19.0509157 L63.2924703,16.5659029 L61.017121,16.5659029 L61.017121,19.0509157 L59.3532718,19.0509157 L59.3532718,20.7541266 L61.017121,20.7541266 L61.017121,26.5617968 C61.017121,28.5163012 61.8988189,29.4935534 63.6479937,29.4935534 C64.1315054,29.4935534 64.6292381,29.4237497 65.1554126,29.2701815 L65.1554126,27.483206 C64.8852149,27.5530098 64.6150171,27.5809313 64.3590403,27.5809313 C63.9466333,27.594892 63.6764355,27.5111275 63.5200053,27.3296378 Z M73.3040074,19.0648764 L75.5793567,19.0648764 L75.5793567,29.3120637 L73.3040074,29.3120637 L73.3040074,19.0648764 Z M88.4919641,26.0173277 L86.3446032,19.0648764 L83.8843818,19.0648764 L87.453836,29.2562207 L87.1267545,30.1357477 C86.9561033,30.6383346 86.7285684,30.9873532 86.4299288,31.1967644 C86.1455102,31.4061756 85.7046612,31.5178615 85.1358239,31.5178615 L84.7091959,31.4899401 L84.7091959,33.2769155 C85.107382,33.3886015 85.4771263,33.4444444 85.8042078,33.4444444 C87.2831848,33.4444444 88.3355339,32.5788782 88.9612549,30.8617065 L93,19.0648764 L90.5682204,19.0648764 L88.4919641,26.0173277 Z M80.0447298,15.4909255 C79.4190087,16.1051983 79.1061482,16.9847253 79.1061482,18.1295064 L79.1061482,19.0648764 L77.5560664,19.0648764 L77.5560664,20.7680874 L79.1061482,20.7680874 L79.1061482,29.3120637 L81.3814975,29.3120637 L81.3814975,20.7680874 L83.4435328,20.7680874 L83.4435328,19.0648764 L81.3814975,19.0648764 L81.3814975,18.1574279 C81.3814975,17.0266075 81.921893,16.4681777 83.0169048,16.4681777 C83.3439863,16.4681777 83.6426259,16.4960992 83.8843818,16.5379814 L83.9412655,14.7370452 C83.4861956,14.6253593 83.0737886,14.5695163 82.6613815,14.5695163 C81.5521487,14.5555556 80.6704508,14.8766527 80.0447298,15.4909255 Z M75.5793567,14.5555556 L75.5793567,16.5659029 L73.3040074,16.5659029 L73.3040074,14.5555556 L75.5793567,14.5555556 Z"
              id="Combined-Shape"
            />
          </g>
          <path
            d="M47,18.7088317 L47,13.0811682 L48.9325768,13.0811682 C49.6205775,13.0811682 50.187461,13.3008356 50.6332443,13.7401769 C51.0790276,14.1795182 51.3019159,14.7431808 51.3019159,15.4311816 L51.3019159,16.3626836 C51.3019159,17.0532611 51.0790276,17.6169237 50.6332443,18.0536882 C50.187461,18.4904527 49.6205775,18.7088317 48.9325768,18.7088317 L47,18.7088317 Z M48.1286248,13.9508278 L48.1286248,17.8430374 L48.8745995,17.8430374 C49.2791542,17.8430374 49.5960936,17.70647 49.8254272,17.4333311 C50.0547608,17.1601922 50.1694259,16.8033133 50.1694259,16.3626836 L50.1694259,15.4234513 C50.1694259,14.9879751 50.0547608,14.6336729 49.8254272,14.360534 C49.5960936,14.0873952 49.2791542,13.9508278 48.8745995,13.9508278 L48.1286248,13.9508278 Z M56.6358277,16.2351335 L54.3051401,16.2351335 L54.3051401,17.8430374 L57.0300734,17.8430374 L57.0300734,18.7088317 L53.1765153,18.7088317 L53.1765153,13.0811682 L57.0223431,13.0811682 L57.0223431,13.9508278 L54.3051401,13.9508278 L54.3051401,15.3654739 L56.6358277,15.3654739 L56.6358277,16.2351335 Z M59.84004,16.6680307 L59.84004,18.7088317 L58.7114151,18.7088317 L58.7114151,13.0811682 L60.9532042,13.0811682 C61.5999764,13.0811682 62.1088832,13.2460798 62.4799398,13.5759079 C62.8509964,13.905736 63.0365219,14.3399172 63.0365219,14.8784646 C63.0365219,15.417012 62.8509964,15.8499049 62.4799398,16.1771562 C62.1088832,16.5044075 61.5999764,16.6680307 60.9532042,16.6680307 L59.84004,16.6680307 Z M59.84004,15.7983711 L60.9532042,15.7983711 C61.2675716,15.7983711 61.5059203,15.7126944 61.6682576,15.5413384 C61.8305948,15.3699824 61.9117623,15.1516034 61.9117623,14.8861949 C61.9117623,14.6156328 61.831239,14.3921003 61.6701902,14.2155908 C61.5091413,14.0390812 61.2701484,13.9508278 60.9532042,13.9508278 L59.84004,13.9508278 L59.84004,15.7983711 Z M65.9856341,17.8430374 L68.4902535,17.8430374 L68.4902535,18.7088317 L64.8570092,18.7088317 L64.8570092,13.0811682 L65.9856341,13.0811682 L65.9856341,17.8430374 Z M74.4271293,16.4013351 C74.4271293,17.0919126 74.2087503,17.6626613 73.7719858,18.1135981 C73.3352213,18.5645349 72.7676936,18.79 72.0693857,18.79 C71.3762314,18.79 70.813213,18.5645349 70.3803136,18.1135981 C69.9474142,17.6626613 69.7309678,17.0919126 69.7309678,16.4013351 L69.7309678,15.3886649 C69.7309678,14.7006641 69.9467701,14.1305597 70.378381,13.6783344 C70.809992,13.2261092 71.3723662,13 72.0655205,13 C72.7638284,13 73.3320003,13.2261092 73.7700532,13.6783344 C74.2081061,14.1305597 74.4271293,14.7006641 74.4271293,15.3886649 L74.4271293,16.4013351 Z M73.2985045,15.3809346 C73.2985045,14.9428816 73.1889929,14.583426 72.9699665,14.3025567 C72.75094,14.0216875 72.449461,13.881255 72.0655205,13.881255 C71.68158,13.881255 71.3839662,14.0210433 71.1726701,14.3006242 C70.961374,14.580205 70.8557275,14.9403049 70.8557275,15.3809346 L70.8557275,16.4013351 C70.8557275,16.8471184 70.9626623,17.2104392 71.1765352,17.4913084 C71.3904081,17.7721776 71.688022,17.9126101 72.0693857,17.9126101 C72.455903,17.9126101 72.7573819,17.7721776 72.9738316,17.4913084 C73.1902813,17.2104392 73.2985045,16.8471184 73.2985045,16.4013351 L73.2985045,15.3809346 Z M78.0642388,15.6746862 L78.0874297,15.6746862 L79.3165485,13.0811682 L80.5533976,13.0811682 L78.6208209,16.7298731 L78.6208209,18.7088317 L77.4960612,18.7088317 L77.4960612,16.6718958 L75.5982708,13.0811682 L76.83512,13.0811682 L78.0642388,15.6746862 Z M85.0524363,17.2400734 C85.0524363,17.0236237 84.9757782,16.8496936 84.8224597,16.7182777 C84.6691412,16.5868618 84.4005157,16.4631781 84.0165752,16.3472229 C83.3466119,16.1539643 82.8402819,15.9246342 82.4975699,15.6592256 C82.1548579,15.3938171 81.9835044,15.0304963 81.9835044,14.5692523 C81.9835044,14.1080084 82.1799811,13.7311597 82.5729404,13.4386949 C82.9658996,13.1462302 83.4677203,13 84.0784176,13 C84.6968453,13 85.2005986,13.1642674 85.5896927,13.4928071 C85.9787867,13.8213468 86.166889,14.2265396 86.1540051,14.7083979 L86.1462748,14.7315888 L85.0524363,14.7315888 C85.0524363,14.4713338 84.9654712,14.260685 84.7915385,14.0996362 C84.6176057,13.9385873 84.3734593,13.8580641 84.0590919,13.8580641 C83.7576084,13.8580641 83.5244131,13.9250594 83.3594991,14.0590521 C83.1945851,14.1930447 83.1121293,14.3643981 83.1121293,14.5731175 C83.1121293,14.7637993 83.2003827,14.9203365 83.3768923,15.0427336 C83.5534018,15.1651308 83.8581017,15.2946121 84.2910011,15.4311816 C84.9120055,15.6038259 85.3822612,15.8318677 85.7017821,16.1153137 C86.0213031,16.3987597 86.1810612,16.7710991 86.1810612,17.2323431 C86.1810612,17.7142013 85.9910263,18.0942709 85.610951,18.3725634 C85.2308757,18.6508558 84.7290549,18.79 84.1054737,18.79 C83.4921996,18.79 82.9575254,18.6321744 82.501435,18.3165187 C82.0453446,18.0008629 81.8237447,17.5583072 81.8366286,16.9888384 L81.8443589,16.9656475 L82.9420625,16.9656475 C82.9420625,17.3006291 83.0444881,17.5460639 83.2493422,17.7019592 C83.4541964,17.8578546 83.7395707,17.935801 84.1054737,17.935801 C84.4121108,17.935801 84.6465944,17.8726708 84.8089317,17.7464085 C84.9712689,17.6201462 85.0524363,17.4513695 85.0524363,17.2400734 Z M90.985447,18.7088317 L90.985447,13.0811682 L92.8871025,13.0811682 C93.5493354,13.0811682 94.0659725,13.2100054 94.4370291,13.4676836 C94.8080857,13.7253618 94.9936112,14.1105849 94.9936112,14.6233645 C94.9936112,14.8836194 94.9246833,15.1161705 94.7868255,15.3210247 C94.6489676,15.5258788 94.4486258,15.6798393 94.1857941,15.7829105 C94.5233525,15.8550604 94.7765175,16.009665 94.9452967,16.246729 C95.114076,16.4837929 95.1984643,16.758216 95.1984643,17.0700066 C95.1984643,17.6085541 95.020669,18.0163237 94.6650731,18.2933277 C94.3094772,18.5703318 93.8057239,18.7088317 93.1537981,18.7088317 L90.985447,18.7088317 Z M92.1140718,16.2196729 L92.1140718,17.8430374 L93.1537981,17.8430374 C93.4527048,17.8430374 93.6807466,17.7779746 93.8379303,17.6478471 C93.995114,17.5177196 94.0737046,17.3251081 94.0737046,17.0700066 C94.0737046,16.794291 94.0067093,16.5836422 93.8727166,16.438054 C93.738724,16.2924659 93.527431,16.2196729 93.2388315,16.2196729 L92.1140718,16.2196729 Z M92.1140718,15.4389119 L92.925754,15.4389119 C93.2272375,15.4389119 93.4591444,15.37707 93.6214817,15.2533845 C93.7838189,15.129699 93.8649863,14.9493269 93.8649863,14.712263 C93.8649863,14.452008 93.7831747,14.2600407 93.6195491,14.1363551 C93.4559234,14.0126696 93.211777,13.9508278 92.8871025,13.9508278 L92.1140718,13.9508278 L92.1140718,15.4389119 Z M98.5109009,15.6746862 L98.5340918,15.6746862 L99.7632106,13.0811682 L101.00006,13.0811682 L99.067483,16.7298731 L99.067483,18.7088317 L97.9427233,18.7088317 L97.9427233,16.6718958 L96.0449329,13.0811682 L97.281782,13.0811682 L98.5109009,15.6746862 Z"
            id="DEPLOYS-BY"
            fill="#BCBCBC"
          />
        </g>
      </g>
    </svg>
  </a>
);

export default NetlifyLogo;
