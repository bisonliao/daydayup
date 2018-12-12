CREATE TABLE `t_first_level_service` (
  `first_level_service_name` char(24) NOT NULL,
  `type` char(20) NOT NULL DEFAULT 'standard',
  PRIMARY KEY (`first_level_service_name`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

CREATE TABLE `t_second_level_service` (
  `second_level_service_name` char(24) NOT NULL,
  `first_level_service_name` char(24) NOT NULL,
  `dev_lang` enum('java','c++') NOT NULL,
  `type` char(20) NOT NULL DEFAULT 'standard',
  PRIMARY KEY (`second_level_service_name`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

CREATE TABLE `t_staff` (
  `staff_name` char(20) NOT NULL,
  `staff_phone` char(20) DEFAULT '',
  `password` char(36) NOT NULL,
  `salt` char(8) NOT NULL DEFAULT 'abcdef01',
  PRIMARY KEY (`staff_name`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
