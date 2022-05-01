---
title: "AWS re:Invent 2018 新機能・新サービス纏め"
date: "2018-12-16"
tags:
  - "AWS"
thumbnail: "aws/reinvent-2018-service-summary/images/thumbnail.png"
---

# AWS re:Invent 2018 新機能・新サービス纏め

## ML

### 新サービス

* [AWS RoboMaker](https://aws.amazon.com/jp/blogs/news/aws-robomaker-develop-test-deploy-and-manage-intelligent-robotics-apps/)
    * ロボットを制御するためのアプリケーション開発をサポートするプラットフォームです。シミュレータによりロボットの動作を確認した上で、実機にデプロイすることが可能です。
* [AWS DeepRacer](https://aws.amazon.com/jp/blogs/aws/aws-deepracer-go-hands-on-with-reinforcement-learning-at-reinvent/)
    * SageMakerやRoboMakerで作成した自動運転用のモデルで制御されるラジコンカーです。
* [Amazon Personalize](https://aws.amazon.com/jp/blogs/news/amazon-personalize-real-time-personalization-and-recommendation-for-everyone/)
    * AutoMLとして学習させるリコメンデーションエンジンです。独自のモデルを簡単に作成することができます。
* [Amazon Forecast](https://aws.amazon.com/jp/blogs/news/amazon-forecast-time-series-forecasting-made-easy/)
    * 簡単に時系列予測用のモデルが作成出来るサービスです。
* [Amazon Textract](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-amazon-textract/)
    * OCR+物体検出で構造化されたデータとして読み込む事ができるスマートOCRです。現在英語版のみです。

### 新機能

* [SageMaker Neo](https://qiita.com/akatsukaha/items/9c90937a20da60ff76b9)
    * SageMakerはAWSのML用プラットフォームサービスです。SageMaker Neoとして、SageMakerの外部で作ったモデルを最適化できる機能が追加されました。最適化されたモデルの実行が早くなります。
* [Amazon Comprehend Medical](https://aws.amazon.com/jp/blogs/news/amazon-comprehend-medical-jp/?sc_channel=sm&sc_publisher=TWITTER&sc_country=Japan&sc_geo=JAPAN&sc_outcome=awareness&trk=_TWITTER&sc_content=amazon-comprehend-medical-jp&sc_category=Amazon+Comprehend&linkId=60264170)
    * 症例報告のようなナラティブから医薬品や有害事象、患者の年齢などのエンティティ抽出を行うことができるAPIです。MedNLP(医療言語処理)と呼ばれる分野のサービスです。現在は英語のみです。
* [Amazon Translateで独自の用語の指定が可能に](https://aws.amazon.com/jp/blogs/machine-learning/introducing-amazon-translate-custom-terminology/?sc_channel=sm&sc_publisher=TWITTER&sc_country=Global&sc_geo=GLOBAL&sc_outcome=awareness&trk=AWS_reInvent_2018_launch_Translate_Custom_Glossary_TWITTER&sc_content=AWS_reInvent_2018_launch_Translate_Custom_Glossary&sc_category=Amazon+Translate&linkId=60253348)
    * 翻訳用APIです。独自の用語(対訳)が指定できるようになりました。Google Translationには無い機能(ただし、GCP AutoMLでは可能)なのでかなり便利になると思います。
* [Amazon Elastic Inference](https://aws.amazon.com/jp/blogs/aws/amazon-elastic-inference-gpu-powered-deep-learning-inference-acceleration/)
    * 推論専用チップである`AWS Inferentia`をEC2インスタンスにアタッチできます。
* [SageMaker Ground Truth](https://aws.amazon.com/jp/blogs/aws/amazon-sagemaker-ground-truth-build-highly-accurate-datasets-and-reduce-labeling-costs-by-up-to-70/)
    * ラベルデータ作成用のプラットフォームです。画像分類、物体検出(BoundingBox/Semantic Segmentation)、テキスト分類用のラベルデータ作成を行う事ができます。ラベルデータ作成の外注も可能です。
    * [AWS SageMaker Ground Truthでテキストのラベリングを試してみる](https://www.inoue-kobo.com/aws/sagemaker-ground-truth/index.html)
* [Marketplace for machine learning](https://aws.amazon.com/jp/blogs/news/new-machine-learning-algorithms-and-model-packages-now-available-in-aws-marketplace/)
    * Marketplaceで学習済みモデルを購入・販売することができます。
* [Amazon SageMaker RL](https://aws.amazon.com/jp/blogs/aws/amazon-sagemaker-rl-managed-reinforcement-learning-with-amazon-sagemaker/)
    * SageMakerに対して強化学習をサポートするための機能を追加したものです。シミュレータなどが付属します。

### その他

* [AWS Inferentia](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-aws-inferentia/)
    * AWSが独自に開発した推論専用チップです。

## Network

### 新サービス

* [AWS Ground Station](https://aws.amazon.com/jp/blogs/news/aws-ground-station-ingest-and-process-data-from-orbiting-satellites/)
    * 衛星通信のための地上の基地局をクラウドとして利用できるサービスです。災害などで地上のネットワークが停止しても(地上局まで通信できれば)通信が可能です。

### 新機能

* [AWS Global Accelerator](https://aws.amazon.com/jp/blogs/aws/new-aws-global-accelerator-for-availability-and-performance/)
    * ロードバランサーです。Globalの名前の通りRegion別にエンドポイントを設定することができます。なお、固定IPを割り振ることができます。
* [AWS Transit Gateway](https://aws.amazon.com/jp/blogs/news/new-aws-transit-gateway/)
    * 今までVPCはPeerでしか接続ができませんでしたが、Transit Gatewayでスター型のトポロジーが可能になります。複数のVPC管理している人にとっては待望の機能です。

## Control

### 新サービス

* [AWS Resource Access Manager](https://aws.amazon.com/jp/blogs/news/new-aws-resource-access-manager-cross-account-resource-sharing/)
    * 複数のAWSアカウントでAWSリソースを共有するためのサービスです。AWSのベストプラクティスとして組織や開発・本番などの用途別にAWSアカウントを分ける事が推奨されていますが、一部のAWSリソースを共有したいこともあり(今までも可能でしたが設定が面倒でした)、相当便利になると思います。
* [AWS Control Tower (Preview)](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-aws-control-tower/)
    * 複数のAWSアカウントにおいて、セキュリティ設定の統制を行うためのサービスです。
* [AWS Security Hub](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-aws-security-hub/)
    * Security関連のAWSサービスが提供することを纏めて確認するためのサービスです。
* [AWS App Mesh](https://aws.amazon.com/jp/blogs/news/reinvent2018-aws-app-mesh/)
    * マイクロサービス用監視サービスです。

## Storage

### 新サービス

* [AWS DataSync](https://aws.amazon.com/jp/blogs/aws/new-aws-datasync-automated-and-accelerated-data-transfer/)
    * オンプレのストレージからS3/EFSにファイルを転送するためのサービスです。
* [Amazon FSx for Windows File Server](https://aws.amazon.com/jp/blogs/aws/new-amazon-fsx-for-windows-file-server-fast-fully-managed-and-secure/)
    * EFSのWindowsサーバ版です。SMB/CIFSでアクセスが可能です。
* [Amazon FSx for Lustre](https://aws.amazon.com/jp/blogs/aws/new-amazon-fsx-for-lustre/)
    * 分散ファイルシステムを採用したファイルサーバです。機械学習などの用途で使用される想定です。

### 新機能

* [AWS Transfer for SFTP](https://aws.amazon.com/jp/blogs/aws/new-aws-transfer-for-sftp-fully-managed-sftp-service-for-amazon-s3/)
    * S3に対してSFTP/SCPによるファイル転送が可能になります。マネージドですがヘッドサーバが作成されるようですので、追加のコストがかかります。
* [EFS Infrequent Access Storage Class](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/coming-soon-amazon-efs-infrequent-access-storage-class/)
    * EFSはフルマネージドなファイルサーバです。Infrequent Access Storage Classが追加になったことにより、使用頻度の低いファイルを格納するストレージのコストを抑えることが可能になりました。ライフサイクルポリシーもサポートされるので、自動的なコスト最適化が可能になります。
* [S3 Intelligent-Tiering](https://aws.amazon.com/jp/blogs/news/new-automatic-cost-optimization-for-amazon-s3-via-intelligent-tiering/)
    * S3には用途に応じた複数のストレージクラスがあり、それぞれでコストが異なります。Intelligent-Tieringを使用すると、アクセス頻度に応じて自動的に最適なストレージクラスに変更してくれるようになります。ライフサイクルポリシーのようなルールベースでは無く、機械学習による推論により最適化が行われます。
* [Amazon S3 Object Lock](https://aws.amazon.com/jp/blogs/architecture/amazon-s3-amazon-s3-glacier-launch-announcements-for-archival-workloads/?sc_channel=sm&sc_publisher=TWITTER&sc_country=Global&sc_geo=GLOBAL&sc_outcome=awareness&trk=AWS_reInvent_2018_launch_S3_Object_Lock_TWITTER&sc_content=AWS_reInvent_2018_launch_S3_Object_Lock&sc_category=Amazon+Simple+Storage+Service+(Amazon+S3)&linkId=60210483)
    * S3上のオブジェクトをロックし、変更や削除ができないようにする機能です。バージョニングは有効なので、個人的にはそこもロックできるようになって欲しいです。今後に期待。
* [AWS SnowballEdge](https://aws.amazon.com/jp/blogs/aws/coming-soon-snowball-edge-with-more-compute-power-and-a-gpu/)
* [Amazon S3 Batch Operations(Preview)](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/s3-batch-operations/)
    * S3上のオブジェクトに対して一括処理が可能になります。今までも可能でしたが、オブジェクトのリストを取得して処理するという普通のプログラミングの流れでした。このため、大量のオブジェクトを処理するために色々な考慮が必要でしたが、バッチ処理機能を利用することで、処理の実行についてはAWSがジョブ管理してくれるようになります。
* [Glacier Deep Archive](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-glacier-deep-archive/)
    * Glacierは低頻度アクセスファイル格納用の低コストストレージです。法令対応などで長期保存が必要なファイルの保存に適しています。Glacier Deep Archiveではさらに安くなっています。

## Development

### 新サービス

* [AWS Amplify Console](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/announcing-aws-amplify-console/)
    * AmplifyはAWSが進めているWebアプリケーション開発用のフレームワークです。今回追加されたAWS Amplify Consoleでは、Webアプリケーションのビルド～デプロイまでのCI/CD環境に加え、公開用のS3の用意やドメインのセットアップなどのリリースのために必要な設定を一括で行う事ができます。
    * SPAのように静的コンテンツで構成されたWebアプリケーションであれば、インフラを一切考慮することなくリリースが完了します。

### 新機能

* [AWS CodeDeploy/AWS CodePipelineでAWS Fargate/Amazon ECSのCI/CDをサポート](https://aws.amazon.com/jp/blogs/devops/build-a-continuous-delivery-pipeline-for-your-container-images-with-amazon-ecr-as-source/)
    * CodeDeployは開発パイプラインを実現するためのサービスです。FargateやECSを開発パイプラインにのせてCI/CDすることが可能になります。
* [AWS Lambda Layer](https://aws.amazon.com/jp/blogs/news/new-for-aws-lambda-use-any-programming-language-and-share-common-components/)
    * AWS Lambdaを共通コンポーネントとして階層化するための機能です。Lambdaをライブラリのように扱う事ができます。
    * [[検証]LambdaのLayer機能を早速試してみた](https://dev.classmethod.jp/cloud/aws/lambda-layer-first-action/)
* [AWS Lambda Runtime API](https://aws.amazon.com/jp/blogs/news/new-for-aws-lambda-use-any-programming-language-and-share-common-components/)
    * AWS Lambdaで様々なプログラミング言語を使用することができるようになりました。
* [AWS Toolkits for PyCharm, IntelliJ (Preview), and Visual Studio Code (Preview)](https://aws.amazon.com/jp/blogs/aws/new-aws-toolkits-for-pycharm-intellij-preview-and-visual-studio-code-preview/)
    * SAM(Serverless Application Model)について、PyCharm, IntelliJ, Visual Studio Code用のプラグインが追加されました。
* [Compute, Database, Messaging, Analytics, and Machine Learning Integration for AWS Step Functions](https://aws.amazon.com/jp/blogs/aws/new-compute-database-messaging-analytics-and-machine-learning-integration-for-aws-step-functions/)
    * StepFunctionsはサーバレスのWorkflowを実装するためのサービスです。今回、以下のAWSリソースをStepFunctionsから直接操作できるようになりました。
        * DynamoDB
        * AWS Batch
        * Amazon ECS
        * Amazon SNS
        * Amazon SQS
        * AWS Glue
        * Amazon SageMaker
* [AWS Cloud Map](https://aws.amazon.com/jp/blogs/news/new-application-integration-with-aws-cloud-map-for-service-discovery/)
    * サービスディスカバリサービスです。

## Operations

### 新機能

* [Amazon CloudWatch Logs Insightsでログの分析が可能に](https://aws.amazon.com/jp/blogs/aws/new-amazon-cloudwatch-logs-insights-fast-interactive-log-analytics/)
    * CloudWatch Logsでログは集中管理するのが定番ですが、あまり分析的な機能はなかったのでElasticsearchなどを併用するのが定石でした。これでログ分析についてはElasticsearch使わなくてもよくなるかも?

## Database

### 新サービス

* [AWS Lake Formation](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-aws-lake-formation/)
    * データレイクを簡単にセットアップするためのサービスです。
* [Amazon Timestream](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-amazon-timestream/)
    * サーバーレスな時系列データベースです。
* [Amazon Quantum Ledger Database](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-keynote-amazon-quantum-ledger-database/)
    * フルマネージドな台帳データベースです。完全追記型のデータベースになります。

### 新機能

* [DynamoDBのトランザクションサポート](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/announcing-amazon-dynamodb-support-for-transactions/)
    * DynamoDBはサーバレスのスキーマレスデータベースサービスです。トランザクションが追加されたことにより、RDB感覚での処理に近づきました。結構衝撃の機能追加です。
* [Amazon DynamoDB on Demand](https://aws.amazon.com/jp/blogs/aws/amazon-dynamodb-on-demand-no-capacity-planning-and-pay-per-request-pricing/)
    * 面倒だったキャパシティプランニングが自動になりました。こっちも衝撃の機能追加です。
* [Preview of Amazon Aurora PostgreSQL Serverless](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/sign-up-for-the-preview-of-amazon-aurora-postgresql-serverless/)
    * Aurora Serverlessはデータベースインスタンスを常時起動することなく、リクエストに応じて自動的に起動・処理・停止する仕組みです。MySQL版はリリース済みでしたが、新しくPostgreSQL版も利用できるようになりました。

## Data Analysis

### 新機能

* [Amazon QuickSightのダッシュボードを独自のアプリケーションに埋め込み可能に](https://aws.amazon.com/jp/blogs/big-data/embed-interactive-dashboards-in-your-application-with-amazon-quicksight/)
    * QuickSightはBIのためのサービスです。ダッシュボードを独自に組み込めると、グラフやチャートを表示するアプリで個別に開発が不要になって便利です。
* [Amazon QuickSight announces ML Insights(Preview)](https://aws.amazon.com/jp/blogs/big-data/amazon-quicksight-announces-ml-insights-in-preview/)
    * まぁ、このあたりは定番ですね。

## Package Business

### 新機能

* [Private Marketplace](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/awsmarketplace-makes-it-easier-to-govern-software-procurement-with-privatemarketplace/)
    * Marketplaceは仮想マシンのイメージ(AMI)を販売・購入するためのサービスです。AMIを社内にだけ公開とかそういうことができるようになりました。
* [AWS Marketplace for Containers](https://aws.amazon.com/jp/blogs/news/reinvent-2018-aws-marketplace-for-containers/)
    * これは以外と便利です。AMIを直接コンテナとしてデプロイできます。

## Instance

### 新機能

* [Elastic Fabric Adapter](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/introducing-elastic-fabric-adapter/)
    * 高速なネットワークインターフェイスです。GPUクラスタなどでの利用を想定していると思われます。
* [A1](https://aws.amazon.com/jp/blogs/news/new-ec2-instances-a1-powered-by-arm-based-aws-graviton-processors/)
    * AWSが独自に設計したCPUを使うインスタンスです。費用対効果に優れます。
* [C5n](https://aws.amazon.com/jp/blogs/news/new-c5n-instances-with-100-gbps-networking/)
    * 高いネットワーク帯域幅を備えたC系インスタンスです。HPC用です。
* [P3dn](https://aws.amazon.com/jp/ec2/instance-types/p3/)
    * 高いネットワーク帯域幅を備えたP系インスタンスです。基本的に機械学習用です。
* [Amazon Lightsail Now Provides an Upgrade Path to EC2](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/amazon-lightsail-now-provides-an-upgrade-path-to-ec2/)
    * Amazon Lightsailは簡単にVPSとしてサーバを起動出来るサービスです。Lightsailで作成したインスタンスを、簡単にEC2に移行できるようになりました。試験的にLightsailで作り、EC2で本番運用することが可能です。
* [Hibernate Your EC2 Instances](https://aws.amazon.com/jp/blogs/aws/new-hibernate-your-ec2-instances/)
    * EC2インスタンスをハイバネーションで停止出来ます。ハイバネーションなので停止から復帰指示にプロセスが生きた状態で再開できます。ElasticCacheとかでも実装されないかなぁ。

## IoT

### 新サービス

* [AWS IoT SiteWise](https://aws.amazon.com/jp/iot-sitewise/)
* [AWS IoT Device Tester](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/aws-iot-device-tester-now-available/)
* [AWS IoT Events](https://aws.amazon.com/jp/about-aws/whats-new/2018/11/introducing-aws-iot-events-now-available-in-preview/)

## Training

### 新機能

* [AWS Certified Machine Learning](https://aws.amazon.com/jp/blogs/machine-learning/amazons-own-machine-learning-university-now-available-to-all-developers/)
    * AWSの機械学習エンジニア用トレーニングコンテンツです。現在は英語のみです。

## Other

### 新サービス

* [AWS Managed Blockchain](https://aws.amazon.com/jp/blogs/news/reinvent-2018-andy-jassy-keynote-amazon-managed-blockchain/)
    * ブロックチェーンサービスです。ブロックチェーンに必要なインフラをマネージドな環境で利用することができます。
* [Amazon Outposts](https://aws.amazon.com/jp/outposts/)
    * オンプレでAWSの機能一式を運用することが可能です。

## OSS

* [Dynamic training](https://aws.amazon.com/jp/blogs/machine-learning/introducing-dynamic-training-for-deep-learning-with-amazon-ec2/)
    * DeepLearningにおいて学習用インスタンスを動的に増減させるためのミドルウェアです。EC2インスタンスなどで利用します。
* [Firecracker](https://aws.amazon.com/jp/blogs/news/firecracker-lightweight-virtualization-for-serverless-computing/)
    * FargateやLambdaを実現しているコンテナ管理の仕組みです。Dockerみたいなもんです。

## Media

* [AWS Elemental MediaConnect](https://aws.amazon.com/jp/blogs/aws/new-aws-elemental-mediaconnect-for-ingestion-and-distribution-of-video-in-the-cloud/)

## 参考文献

* [【速報】AWS re:Invent 2018 Keynote 1日目で発表された新サービスまとめ](https://dev.classmethod.jp/cloud/aws/aws-reinvent-2018-keynote-day-1/)
* [AWS re:Invent 製品発表](https://aws.amazon.com/jp/new/reinvent/?hp=r)