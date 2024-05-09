// SPDX-License-Identifier: MIT
// Copyright contributors to the kepler.gl project

import React, {Component} from 'react';
import Base, {BaseProps} from './base';

export default class Logout extends Component<Partial<BaseProps>> {
  static defaultProps = {
    height: '16px',
    predefinedClassName: 'data-ex-icons-logout'
  };

  render() {
    return (
      <Base {...this.props}>
        <polygon points="27.1024306 41.981391 23.3612739 45.7225477 10 32.3612739 23.3612739 19 27.1024306 22.7411567 20.1545681 29.6890191 46.0754395 29.6890191 46.0754395 35.0335286 20.1545681 35.0335286" />
        <path d="M50.7560765,8 L13.3445095,8 C10.4050293,8 8,10.4050293 8,13.3445095 L8,24.0335286 L13.3445095,24.0335286 L13.3445095,13.3445095 L50.7560765,13.3445095 L50.7560765,50.7560765 L13.3445095,50.7560765 L13.3445095,40.0670573 L8,40.0670573 L8,50.7560765 C8,53.6955566 10.4050293,56.1005859 13.3445095,56.1005859 L50.7560765,56.1005859 C53.6955566,56.1005859 56.1005859,53.6955566 56.1005859,50.7560765 L56.1005859,13.3445095 C56.1005859,10.4050293 53.6955566,8 50.7560765,8 Z" />
      </Base>
    );
  }
}